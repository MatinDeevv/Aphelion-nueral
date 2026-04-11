"""HMM-based regime detection for the Phase 7 mixture-of-experts stack."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import pickle
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

from .features import RegimeFeatureExtractor


def _build_sequence_model(
    *,
    n_regimes: int,
    covariance_type: str,
    n_iter: int,
) -> tuple[Any, str]:
    """Return the best available probabilistic sequence model for regime fitting."""

    try:
        from hmmlearn.hmm import GaussianHMM
    except ModuleNotFoundError:
        GaussianHMM = None

    if GaussianHMM is not None:
        return (
            GaussianHMM(
                n_components=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=7,
            ),
            "hmmlearn_gaussian_hmm",
        )

    try:
        from sklearn.mixture import GaussianMixture
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RegimeDetector.fit() requires hmmlearn or scikit-learn. "
            "Install one of them to fit or load a persisted detector.",
        ) from exc

    return (
        GaussianMixture(
            n_components=n_regimes,
            covariance_type=covariance_type,
            max_iter=n_iter,
            random_state=7,
        ),
        "sklearn_gaussian_mixture",
    )


@dataclass(slots=True)
class RegimeState:
    """Named regime state with a canonical four-regime probability vector."""

    N_REGIMES: ClassVar[int] = 4
    NAMES: ClassVar[list[str]] = ["trending", "mean_reverting", "volatile", "quiet"]

    probs: Tensor
    dominant: str
    confidence: float

    @classmethod
    def from_probs(cls, probs: Tensor) -> "RegimeState":
        """Build a canonical regime state from a probability vector."""

        if probs.dim() != 1 or probs.numel() != cls.N_REGIMES:
            raise ValueError(
                f"Regime probabilities must be shaped [{cls.N_REGIMES}], got {tuple(probs.shape)}",
            )

        normalized = probs.detach().float().cpu()
        if not torch.isfinite(normalized).all():
            normalized = torch.full((cls.N_REGIMES,), 1.0 / cls.N_REGIMES, dtype=torch.float32)
        elif (normalized < 0).any() or normalized.sum().item() <= 0.0:
            normalized = torch.softmax(normalized, dim=-1)
        else:
            normalized = normalized / normalized.sum()

        dominant_index = int(torch.argmax(normalized).item())
        return cls(
            probs=normalized,
            dominant=cls.NAMES[dominant_index],
            confidence=float(normalized[dominant_index].item()),
        )


class RegimeDetector(nn.Module):
    """Online Gaussian-HMM regime classifier over a fixed microstructure feature subset."""

    def __init__(
        self,
        n_regimes: int = 4,
        n_iter: int = 100,
        covariance_type: str = "diag",
        window: int = 60,
    ) -> None:
        super().__init__()
        if n_regimes != RegimeState.N_REGIMES:
            raise ValueError(
                f"Phase 7 currently expects exactly {RegimeState.N_REGIMES} regimes, got {n_regimes}.",
            )
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.window = window
        self.extractor = RegimeFeatureExtractor()
        self._model: Any | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._state_order: tuple[int, ...] = tuple(range(self.n_regimes))
        self._backend: str = "unfitted"

    @property
    def is_fitted(self) -> bool:
        """Return whether the detector has a fitted HMM and standardization statistics."""

        return (
            self._model is not None
            and self._feature_mean is not None
            and self._feature_std is not None
        )

    def fit(self, feature_df: pl.DataFrame) -> "RegimeDetector":
        """Fit the Gaussian HMM on a historical feature dataframe."""

        observations = self.extractor.extract(feature_df)
        if observations.shape[0] < self.n_regimes:
            raise ValueError(
                f"Need at least {self.n_regimes} rows to fit the regime detector, got {observations.shape[0]}.",
            )

        self._feature_mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        std = observations.std(axis=0, dtype=np.float64).astype(np.float32, copy=False)
        self._feature_std = np.maximum(std, 1e-6).astype(np.float32, copy=False)
        standardized = self._standardize(observations)

        model, backend = _build_sequence_model(
            n_regimes=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
        )
        model.fit(standardized)

        self._model = model
        self._backend = backend
        self._state_order = self._infer_state_order(np.asarray(model.means_, dtype=np.float32))
        return self

    def forward(self, features: Tensor) -> RegimeState:
        """Classify one rolling observation window into a soft regime state."""

        if features.dim() != 2:
            raise ValueError(
                "RegimeDetector.forward() expects [window, n_regime_features], "
                f"got {tuple(features.shape)}.",
            )
        if features.size(-1) != len(self.extractor.REGIME_FEATURE_COLS):
            raise ValueError(
                "Unexpected regime feature dimension. "
                f"Expected {len(self.extractor.REGIME_FEATURE_COLS)}, got {features.size(-1)}.",
            )
        if not self.is_fitted:
            return self._uniform_state()

        observation_array = features.detach().float().cpu().numpy().astype(np.float32, copy=False)
        standardized = self._standardize(observation_array)
        raw_probs = self._posterior_probabilities(standardized)
        canonical = raw_probs[:, list(self._state_order)]
        return RegimeState.from_probs(torch.from_numpy(canonical[-1]).float())

    def forward_sequence(self, feature_df: pl.DataFrame) -> list[RegimeState]:
        """Classify an entire historical dataframe into one soft regime state per row."""

        observations = self.extractor.extract(feature_df)
        if observations.shape[0] == 0:
            return []
        if not self.is_fitted:
            return [self._uniform_state() for _ in range(observations.shape[0])]

        standardized = self._standardize(observations)
        raw_probs = self._posterior_probabilities(standardized)
        canonical = raw_probs[:, list(self._state_order)]
        return [RegimeState.from_probs(torch.from_numpy(row).float()) for row in canonical]

    def save(self, path: Path) -> None:
        """Persist the fitted HMM to pickle alongside detector metadata JSON."""

        if not self.is_fitted:
            raise RuntimeError("RegimeDetector.save() called before fit().")

        model_path = self._model_path(path)
        metadata_path = self._metadata_path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with model_path.open("wb") as handle:
            pickle.dump(self._model, handle)

        metadata = {
            "n_regimes": self.n_regimes,
            "n_iter": self.n_iter,
            "covariance_type": self.covariance_type,
            "window": self.window,
            "backend": self._backend,
            "feature_names": list(self.extractor.REGIME_FEATURE_COLS),
            "feature_mean": self._feature_mean.tolist(),
            "feature_std": self._feature_std.tolist(),
            "state_order": list(self._state_order),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RegimeDetector":
        """Load a persisted detector and restore its fitted state."""

        model_path = cls._model_path(path)
        metadata_path = cls._metadata_path(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        detector = cls(
            n_regimes=int(metadata["n_regimes"]),
            n_iter=int(metadata["n_iter"]),
            covariance_type=str(metadata["covariance_type"]),
            window=int(metadata["window"]),
        )
        with model_path.open("rb") as handle:
            detector._model = pickle.load(handle)
        detector._feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
        detector._feature_std = np.asarray(metadata["feature_std"], dtype=np.float32)
        detector._state_order = tuple(int(index) for index in metadata["state_order"])
        detector._backend = str(metadata.get("backend", "loaded"))
        return detector

    def _posterior_probabilities(self, standardized: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities for each timestep in a standardized sequence."""

        assert self._model is not None
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(standardized)
        else:
            _, probs = self._model.score_samples(standardized)
        return np.asarray(probs, dtype=np.float32)

    def _standardize(self, observations: np.ndarray) -> np.ndarray:
        """Apply the stored feature standardization to a feature matrix."""

        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("RegimeDetector standardization statistics are unavailable.")
        return ((observations - self._feature_mean) / self._feature_std).astype(np.float32, copy=False)

    def _uniform_state(self) -> RegimeState:
        """Return a uniform soft regime state when the detector is unfitted."""

        uniform = torch.full((self.n_regimes,), 1.0 / self.n_regimes, dtype=torch.float32)
        return RegimeState.from_probs(uniform)

    def _infer_state_order(self, standardized_means: np.ndarray) -> tuple[int, ...]:
        """Infer a canonical raw-state-to-regime mapping from HMM emission means."""

        feature_index = {
            name: index
            for index, name in enumerate(self.extractor.REGIME_FEATURE_COLS)
        }

        def score(state_mean: np.ndarray, regime_name: str) -> float:
            if regime_name == "trending":
                return (
                    state_mean[feature_index["trend_alignment_5_15_60"]]
                    + state_mean[feature_index["path_efficiency_20"]]
                    - state_mean[feature_index["direction_switch_rate_20"]]
                )
            if regime_name == "mean_reverting":
                return (
                    state_mean[feature_index["direction_switch_rate_20"]]
                    - state_mean[feature_index["trend_alignment_5_15_60"]]
                    - state_mean[feature_index["path_efficiency_20"]]
                )
            if regime_name == "volatile":
                return (
                    state_mean[feature_index["realized_vol"]]
                    + state_mean[feature_index["relative_spread"]]
                    + state_mean[feature_index["disagreement_pressure_bps"]]
                    + state_mean[feature_index["volatility_ratio_5_60"]]
                )
            if regime_name == "quiet":
                return (
                    state_mean[feature_index["silence_ratio_20"]]
                    - state_mean[feature_index["tick_rate_hz"]]
                    - state_mean[feature_index["realized_vol"]]
                )
            raise KeyError(regime_name)

        best_order: tuple[int, ...] | None = None
        best_score = float("-inf")
        for candidate in itertools.permutations(range(self.n_regimes)):
            candidate_score = sum(
                score(standardized_means[state_index], regime_name)
                for state_index, regime_name in zip(candidate, RegimeState.NAMES, strict=True)
            )
            if candidate_score > best_score:
                best_score = candidate_score
                best_order = tuple(candidate)

        assert best_order is not None
        return best_order

    @staticmethod
    def _model_path(path: Path) -> Path:
        """Resolve the pickle path used for a persisted regime detector."""

        path = Path(path)
        return path if path.suffix == ".pkl" else path.with_suffix(".pkl")

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        """Resolve the metadata JSON path used for a persisted regime detector."""

        path = Path(path)
        if path.suffix == ".json":
            return path
        if path.suffix == ".pkl":
            return path.with_suffix(".json")
        return path.with_suffix(".json")


__all__ = ["RegimeDetector", "RegimeState"]
