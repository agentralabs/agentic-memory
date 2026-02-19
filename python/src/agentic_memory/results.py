"""Result dataclasses for advanced AgenticMemory queries.

These cover the extended query surface (queries 8-16): text search,
hybrid search, centrality, shortest path, revision analysis, gap
detection, analogy, consolidation, and drift analysis.

All result types are plain dataclasses (mutable) to allow incremental
construction during JSON parsing.  They are *not* part of the core
models module, which contains the immutable Event/Edge/BrainInfo types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ===================================================================
# Text & Hybrid Search
# ===================================================================

@dataclass
class TextMatch:
    """A single text-search hit.

    Attributes:
        node_id: The matching node's ID.
        score: BM25 / TF-IDF relevance score.
        matched_terms: Which query terms matched.
    """
    node_id: int
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class HybridMatch:
    """A single hybrid-search (text + vector) hit.

    Attributes:
        node_id: The matching node's ID.
        combined_score: Fused relevance score.
        text_rank: Rank in the text-only result set (0 = absent).
        vector_rank: Rank in the vector-only result set (0 = absent).
        text_score: Raw text relevance score.
        vector_similarity: Cosine similarity from vector search.
    """
    node_id: int
    combined_score: float
    text_rank: int = 0
    vector_rank: int = 0
    text_score: float = 0.0
    vector_similarity: float = 0.0


# ===================================================================
# Graph Algorithms
# ===================================================================

@dataclass
class PathResult:
    """Result of a shortest-path query between two nodes.

    Attributes:
        path: Ordered list of node IDs from source to target.
        edges: Edge metadata for each hop.
        cost: Total path cost (sum of edge weights).
        found: Whether a path was found.
    """
    path: list[int]
    edges: list[dict]  # type: ignore[type-arg]
    cost: float
    found: bool = True


# ===================================================================
# Revision & Gap Analysis
# ===================================================================

@dataclass
class RevisionReport:
    """Result of revising the knowledge graph after new evidence.

    Attributes:
        contradicted: List of nodes whose confidence was reduced.
        weakened: List of nodes partially weakened.
        invalidated_decisions: IDs of decisions that lost support.
        total_affected: Total nodes affected by the revision.
        cascade: Ordered list of cascade steps.
    """
    contradicted: list[dict]  # type: ignore[type-arg]
    weakened: list[dict]  # type: ignore[type-arg]
    invalidated_decisions: list[int]
    total_affected: int
    cascade: list[dict]  # type: ignore[type-arg]


@dataclass
class GapReport:
    """Result of gap analysis on the knowledge graph.

    Attributes:
        gaps: List of identified knowledge gaps.
        health_score: Overall graph health, 0.0 to 1.0.
        summary: Aggregated gap statistics.
    """
    gaps: list[dict]  # type: ignore[type-arg]
    health_score: float
    summary: dict  # type: ignore[type-arg]


# ===================================================================
# Analogy & Pattern Matching
# ===================================================================

@dataclass
class Analogy:
    """A structural analogy found across sessions.

    Attributes:
        center_id: Hub node of the analogous subgraph.
        sessions: Session IDs participating in the analogy.
        structural_similarity: Graph-topology similarity, 0.0 to 1.0.
        content_similarity: Textual similarity, 0.0 to 1.0.
        combined_score: Weighted combination of structural and content.
        pattern_matches: Detailed node-to-node correspondences.
    """
    center_id: int
    sessions: list[int]
    structural_similarity: float
    content_similarity: float
    combined_score: float
    pattern_matches: list[dict]  # type: ignore[type-arg]


# ===================================================================
# Consolidation & Drift
# ===================================================================

@dataclass
class ConsolidationReport:
    """Result of a consolidation pass over the knowledge graph.

    Attributes:
        actions: List of actions taken (or proposed in dry-run mode).
        deduplicated: Number of duplicate nodes merged.
        contradictions_linked: Number of contradiction edges added.
        inferences_promoted: Number of inferences promoted to facts.
        dry_run: Whether this was a dry-run (no mutations applied).
        backup_path: Path to the backup file (if created).
    """
    actions: list[dict]  # type: ignore[type-arg]
    deduplicated: int = 0
    contradictions_linked: int = 0
    inferences_promoted: int = 0
    dry_run: bool = True
    backup_path: Optional[str] = None


@dataclass
class DriftReport:
    """Result of belief-drift analysis over time.

    Attributes:
        timelines: Per-topic drift timelines.
        stability: Overall belief stability, 0.0 to 1.0.
        likely_to_change: Whether beliefs are trending toward change.
    """
    timelines: list[dict]  # type: ignore[type-arg]
    stability: float
    likely_to_change: bool


# ===================================================================
# Parsing Helpers (internal -- used by Brain to convert CLI JSON output)
# ===================================================================

def parse_text_match(data: dict) -> TextMatch:  # type: ignore[type-arg]
    """Parse a CLI JSON text-search hit into a TextMatch."""
    return TextMatch(
        node_id=data.get("node_id", data.get("id", 0)),
        score=float(data.get("score", 0.0)),
        matched_terms=data.get("matched_terms", data.get("terms", [])),
    )


def parse_hybrid_match(data: dict) -> HybridMatch:  # type: ignore[type-arg]
    """Parse a CLI JSON hybrid-search hit into a HybridMatch."""
    return HybridMatch(
        node_id=data.get("node_id", data.get("id", 0)),
        combined_score=float(data.get("combined_score", data.get("score", 0.0))),
        text_rank=int(data.get("text_rank", 0)),
        vector_rank=int(data.get("vector_rank", 0)),
        text_score=float(data.get("text_score", 0.0)),
        vector_similarity=float(data.get("vector_similarity", data.get("similarity", 0.0))),
    )


def parse_path_result(data: dict) -> PathResult:  # type: ignore[type-arg]
    """Parse a CLI JSON path query result into a PathResult."""
    path = data.get("path", [])
    # Handle both list of ints and list of dicts
    path_ids: list[int] = []
    for item in path:
        if isinstance(item, int):
            path_ids.append(item)
        elif isinstance(item, dict):
            path_ids.append(item.get("id", item.get("node_id", 0)))

    return PathResult(
        path=path_ids,
        edges=data.get("edges", []),
        cost=float(data.get("cost", data.get("total_cost", 0.0))),
        found=data.get("found", len(path_ids) > 0),
    )


def parse_revision_report(data: dict) -> RevisionReport:  # type: ignore[type-arg]
    """Parse a CLI JSON revision result into a RevisionReport."""
    invalidated = data.get("invalidated_decisions", [])
    # Handle both list of ints and list of dicts
    inv_ids: list[int] = []
    for item in invalidated:
        if isinstance(item, int):
            inv_ids.append(item)
        elif isinstance(item, dict):
            inv_ids.append(item.get("id", item.get("node_id", 0)))

    return RevisionReport(
        contradicted=data.get("contradicted", []),
        weakened=data.get("weakened", []),
        invalidated_decisions=inv_ids,
        total_affected=int(data.get("total_affected", 0)),
        cascade=data.get("cascade", []),
    )


def parse_gap_report(data: dict) -> GapReport:  # type: ignore[type-arg]
    """Parse a CLI JSON gap-analysis result into a GapReport."""
    return GapReport(
        gaps=data.get("gaps", []),
        health_score=float(data.get("health_score", data.get("health", 0.0))),
        summary=data.get("summary", {}),
    )


def parse_analogy(data: dict) -> Analogy:  # type: ignore[type-arg]
    """Parse a CLI JSON analogy result into an Analogy."""
    return Analogy(
        center_id=data.get("center_id", data.get("id", 0)),
        sessions=data.get("sessions", []),
        structural_similarity=float(data.get("structural_similarity", 0.0)),
        content_similarity=float(data.get("content_similarity", 0.0)),
        combined_score=float(data.get("combined_score", data.get("score", 0.0))),
        pattern_matches=data.get("pattern_matches", data.get("matches", [])),
    )


def parse_consolidation_report(data: dict) -> ConsolidationReport:  # type: ignore[type-arg]
    """Parse a CLI JSON consolidation result into a ConsolidationReport."""
    return ConsolidationReport(
        actions=data.get("actions", []),
        deduplicated=int(data.get("deduplicated", 0)),
        contradictions_linked=int(data.get("contradictions_linked", 0)),
        inferences_promoted=int(data.get("inferences_promoted", 0)),
        dry_run=data.get("dry_run", True),
        backup_path=data.get("backup_path", None),
    )


def parse_drift_report(data: dict) -> DriftReport:  # type: ignore[type-arg]
    """Parse a CLI JSON drift-analysis result into a DriftReport."""
    return DriftReport(
        timelines=data.get("timelines", []),
        stability=float(data.get("stability", 0.0)),
        likely_to_change=data.get("likely_to_change", False),
    )
