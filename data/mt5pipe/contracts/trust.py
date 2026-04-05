"""
Trust / quality-gate contracts shared across sectors.

Provides the minimal shared vocabulary for trust verdicts.
The full TrustReport and QaCheckResult remain in mt5pipe.truth.models.
"""

from __future__ import annotations

from enum import Enum


class TrustVerdict(str, Enum):
    """
    Coarse trust outcome visible to any sector.

    Maps 1:1 to the existing TrustReport.status values.
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WARNING = "warning"
