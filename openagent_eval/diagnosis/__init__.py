"""Component diagnosis module for OpenAgent Eval.

Provides blame attribution for retrieval, generation, chunking, and recorded
agent tool calls.

Usage::

    from openagent_eval.diagnosis import DiagnosisAnalyzer

    analyzer = DiagnosisAnalyzer()
    report = analyzer.analyze(evaluation_results)

    print(report.blame_summary)
    print(report.recommendations)
"""

from openagent_eval.diagnosis.analyzer import DiagnosisAnalyzer
from openagent_eval.diagnosis.blame import BlameAttribution
from openagent_eval.diagnosis.chunking import ChunkingQualityAnalyzer
from openagent_eval.diagnosis.models import (
    BlameResult,
    BlameTarget,
    ChunkingIssue,
    ComponentScores,
    DiagnosisReport,
    FailureInstance,
    FailureMode,
)

__all__ = [
    "DiagnosisAnalyzer",
    "BlameAttribution",
    "ChunkingQualityAnalyzer",
    "BlameResult",
    "BlameTarget",
    "ChunkingIssue",
    "ComponentScores",
    "DiagnosisReport",
    "FailureInstance",
    "FailureMode",
]
