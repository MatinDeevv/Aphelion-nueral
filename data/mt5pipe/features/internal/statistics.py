"""Shared rolling-stat helpers for machine-native feature families."""

from __future__ import annotations

import math
from collections import Counter
from statistics import pstdev
from typing import Iterable, Sequence


def rolling_shannon_entropy(values: Sequence[object | None], window: int) -> list[float | None]:
    """Return normalized Shannon entropy over a trailing categorical window."""
    results: list[float | None] = []
    if window <= 0:
        return [None] * len(values)

    for idx in range(len(values)):
        if idx + 1 < window:
            results.append(None)
            continue

        sample = [value for value in values[idx + 1 - window : idx + 1] if value is not None]
        if not sample:
            results.append(None)
            continue

        counts = Counter(sample)
        categories = len(counts)
        if categories <= 1:
            results.append(0.0)
            continue

        total = float(sum(counts.values()))
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        results.append(entropy / math.log(categories))

    return results


def rolling_permutation_entropy(
    values: Sequence[float | None],
    window: int,
    *,
    order: int = 3,
    delay: int = 1,
) -> list[float | None]:
    """Return normalized permutation entropy over a trailing numeric window."""
    results: list[float | None] = []
    pattern_span = (order - 1) * delay + 1
    if window <= 0 or order < 2 or delay < 1:
        return [None] * len(values)

    for idx in range(len(values)):
        if idx + 1 < window:
            results.append(None)
            continue

        sample = list(values[idx + 1 - window : idx + 1])
        patterns: list[tuple[int, ...]] = []
        for start in range(0, len(sample) - pattern_span + 1):
            segment = sample[start : start + pattern_span : delay]
            if any(value is None for value in segment):
                continue
            pattern = tuple(order_idx for order_idx, _value in sorted(enumerate(segment), key=lambda item: (item[1], item[0])))
            patterns.append(pattern)

        if not patterns:
            results.append(None)
            continue

        counts = Counter(patterns)
        total = float(sum(counts.values()))
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)

        max_entropy = math.log(math.factorial(order))
        results.append((entropy / max_entropy) if max_entropy > 0 else 0.0)

    return results


def rolling_approximate_entropy(
    values: Sequence[float | None],
    window: int,
    *,
    pattern_size: int = 2,
    tolerance_scale: float = 0.2,
) -> list[float | None]:
    """Return approximate entropy over a trailing numeric window."""
    results: list[float | None] = []
    if window <= 0 or pattern_size < 1:
        return [None] * len(values)

    for idx in range(len(values)):
        if idx + 1 < window:
            results.append(None)
            continue

        sample = [float(value) for value in values[idx + 1 - window : idx + 1] if value is not None]
        if len(sample) < window:
            results.append(None)
            continue

        sigma = pstdev(sample) if len(sample) > 1 else 0.0
        tolerance = tolerance_scale * sigma
        if tolerance <= 0.0:
            results.append(0.0)
            continue

        phi_m = _phi(sample, pattern_size, tolerance)
        phi_m1 = _phi(sample, pattern_size + 1, tolerance)
        results.append(max(phi_m - phi_m1, 0.0))

    return results


def rolling_sample_entropy(
    values: Sequence[float | None],
    window: int,
    *,
    pattern_size: int = 2,
    tolerance_scale: float = 0.2,
) -> list[float | None]:
    """Return sample entropy over a trailing numeric window."""
    results: list[float | None] = []
    if window <= 0 or pattern_size < 1:
        return [None] * len(values)

    for idx in range(len(values)):
        if idx + 1 < window:
            results.append(None)
            continue

        sample = [float(value) for value in values[idx + 1 - window : idx + 1] if value is not None]
        if len(sample) < window:
            results.append(None)
            continue

        sigma = pstdev(sample) if len(sample) > 1 else 0.0
        tolerance = tolerance_scale * sigma
        if tolerance <= 0.0:
            results.append(0.0)
            continue

        matches_m = _count_matches(sample, pattern_size, tolerance, exclude_self=True)
        matches_m1 = _count_matches(sample, pattern_size + 1, tolerance, exclude_self=True)
        if matches_m <= 0 or matches_m1 <= 0:
            results.append(0.0)
            continue
        results.append(max(-math.log(matches_m1 / matches_m), 0.0))

    return results


def signed_run_lengths(values: Iterable[float | None]) -> list[int]:
    """Return the current signed run length for each observation."""
    run_lengths: list[int] = []
    current_sign = 0
    current_length = 0

    for value in values:
        sign = _sign(value)
        if sign == 0:
            current_sign = 0
            current_length = 0
            run_lengths.append(0)
            continue

        if sign == current_sign:
            current_length += 1
        else:
            current_sign = sign
            current_length = 1
        run_lengths.append(current_length * sign)

    return run_lengths


def switch_indicators(values: Iterable[float | None]) -> list[int]:
    """Return 1 when the current sign differs from the previous non-zero sign."""
    indicators: list[int] = []
    previous_sign = 0

    for value in values:
        sign = _sign(value)
        if sign == 0:
            indicators.append(0)
            continue

        indicators.append(1 if previous_sign not in (0, sign) else 0)
        previous_sign = sign

    return indicators


def _phi(sample: Sequence[float], pattern_size: int, tolerance: float) -> float:
    windows = [tuple(sample[idx : idx + pattern_size]) for idx in range(len(sample) - pattern_size + 1)]
    if not windows:
        return 0.0

    phi_sum = 0.0
    for left in windows:
        matches = 0
        for right in windows:
            if _chebyshev_distance(left, right) <= tolerance:
                matches += 1
        probability = matches / len(windows)
        if probability > 0.0:
            phi_sum += math.log(probability)
    return phi_sum / len(windows)


def _count_matches(
    sample: Sequence[float],
    pattern_size: int,
    tolerance: float,
    *,
    exclude_self: bool,
) -> float:
    windows = [tuple(sample[idx : idx + pattern_size]) for idx in range(len(sample) - pattern_size + 1)]
    if len(windows) < 2:
        return 0.0

    matches = 0
    comparisons = 0
    for left_idx, left in enumerate(windows):
        for right_idx, right in enumerate(windows):
            if exclude_self and left_idx == right_idx:
                continue
            comparisons += 1
            if _chebyshev_distance(left, right) <= tolerance:
                matches += 1

    if comparisons == 0:
        return 0.0
    return matches / comparisons


def _chebyshev_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(lval - rval) for lval, rval in zip(left, right, strict=True))


def _sign(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1
