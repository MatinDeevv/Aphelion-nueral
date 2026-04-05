"""
Compiler sector — public boundary module.

This is the ONLY module other sectors may import from the compiler package.
All cross-sector consumers should import from here, never from
``mt5pipe.compiler.models``, ``mt5pipe.compiler.service``, etc. directly.

Re-exports
----------
DatasetSpec : pydantic model
    Compiler input spec describing a dataset build.
ExperimentSpec : pydantic model
    Compiler-owned training/evaluation spec for trusted dataset artifacts.
LineageManifest : pydantic model
    Immutable lineage record for a built artifact.
DatasetCompiler : class
    Orchestrates the full compile pipeline.
ArtifactInspection : dataclass
    Deterministic inspection summary for a compiled artifact.
ArtifactDiff : dataclass
    Deterministic diff summary between two compiled artifacts.
compile_dataset_spec : function
    Module-level convenience for compiling a spec file.
inspect_artifact : function
    Resolve and inspect a dataset artifact by id, alias, or manifest path.
diff_artifacts : function
    Compare two compiled dataset artifacts.
run_experiment_spec : function
    Run a trust-gated experiment and register experiment/model artifacts.
inspect_experiment : function
    Resolve and inspect an experiment artifact by id, alias, or manifest path.
inspect_model : function
    Resolve and inspect a model artifact by id, alias, or manifest path.
"""

from mt5pipe.compiler.models import DatasetSpec, ExperimentSpec, LineageManifest
from mt5pipe.compiler.service import (
    ArtifactDiff,
    ArtifactInspection,
    DatasetCompiler,
    compile_dataset_spec,
    diff_artifacts,
    inspect_artifact,
)
from mt5pipe.compiler.training import (
    ExperimentInspection,
    ExperimentRunResult,
    ModelInspection,
    inspect_experiment,
    inspect_model,
    run_experiment_spec,
)

__all__ = [
    # Models
    "DatasetSpec",
    "ExperimentSpec",
    "LineageManifest",
    # Services
    "ArtifactInspection",
    "ArtifactDiff",
    "ExperimentRunResult",
    "ExperimentInspection",
    "ModelInspection",
    "DatasetCompiler",
    "compile_dataset_spec",
    "inspect_artifact",
    "diff_artifacts",
    "run_experiment_spec",
    "inspect_experiment",
    "inspect_model",
]
