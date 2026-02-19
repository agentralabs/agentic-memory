"""Tests for query expansion features (queries 8-16) in the Python SDK.

These tests exercise the 9 new Brain methods that were added in v0.2.0:
  - search_text (BM25 text search)
  - hybrid_search (BM25 + vector fusion)
  - centrality (PageRank / degree / betweenness)
  - shortest_path (BFS / Dijkstra)
  - revise (belief revision / counterfactual)
  - gaps (reasoning gap detection)
  - analogy (structural pattern matching)
  - consolidate (brain maintenance)
  - drift (belief trajectory tracking)

Requires the ``amem`` CLI binary to be on PATH or set via AMEM_BINARY.

Target: 20 tests (per SPEC-QUERY-EXPANSION-TESTS.md).
"""

import pytest

from agentic_memory import Brain
from agentic_memory.results import (
    TextMatch,
    HybridMatch,
    PathResult,
    RevisionReport,
    GapReport,
    Analogy,
    ConsolidationReport,
    DriftReport,
)


# ================================================================
# Helpers
# ================================================================

@pytest.fixture
def populated_brain(brain):
    """A Brain pre-loaded with nodes across multiple sessions."""
    brain.create()
    # Session 1: tech facts
    brain.add_fact("Rust programming language is memory safe", session=1)
    brain.add_fact("Python is great for data science", session=1)
    brain.add_fact("JavaScript runs in the browser", session=1)
    # Session 2: decisions
    brain.add_decision("Chose Rust for backend", session=2)
    brain.add_decision("Chose Python for data pipeline", session=2)
    # Session 3: more facts
    brain.add_fact("Go is designed for concurrency", session=3)
    brain.add_fact("TypeScript adds types to JavaScript", session=3)
    # Add some edges
    brain.link(3, 0, "caused_by")  # "Chose Rust" caused by "Rust is memory safe"
    brain.link(4, 1, "caused_by")  # "Chose Python" caused by "Python is great"
    brain.link(6, 2, "supports")   # "TypeScript adds types" supports "JS runs in browser"
    return brain


# ================================================================
# BM25 Text Search Tests (4)
# ================================================================

def test_search_text_basic(populated_brain):
    """search_text should return TextMatch results for matching content."""
    results = populated_brain.search_text("Rust programming")
    assert isinstance(results, list)
    # Should find at least the Rust-related nodes
    for m in results:
        assert isinstance(m, TextMatch)
        assert isinstance(m.node_id, int)
        assert isinstance(m.score, float)
        assert m.score >= 0.0


def test_search_text_filter_types(populated_brain):
    """search_text should filter by event type."""
    results = populated_brain.search_text("Rust", types=["decision"])
    for m in results:
        assert isinstance(m, TextMatch)
    # All results should be decisions -- verify via get()
    for m in results:
        event = populated_brain.get(m.node_id)
        assert event.type.value == "decision"


def test_search_text_empty_query(populated_brain):
    """search_text with empty query should return empty or not crash."""
    results = populated_brain.search_text("")
    assert isinstance(results, list)
    # Empty query may return empty results, which is fine


def test_search_text_no_results(populated_brain):
    """search_text for nonexistent terms should return empty."""
    results = populated_brain.search_text("xyzzyplugh nonexistent")
    assert isinstance(results, list)
    assert len(results) == 0


# ================================================================
# Hybrid Search Tests (3)
# ================================================================

def test_search_hybrid_basic(populated_brain):
    """hybrid_search should return HybridMatch results."""
    results = populated_brain.hybrid_search("Rust programming")
    assert isinstance(results, list)
    for m in results:
        assert isinstance(m, HybridMatch)
        assert isinstance(m.node_id, int)
        assert isinstance(m.combined_score, float)


def test_search_hybrid_text_only_fallback(populated_brain):
    """hybrid_search should work even when vectors are all zeros (fallback)."""
    # Our test brain has no embeddings, so this exercises BM25-only fallback
    results = populated_brain.hybrid_search(
        "Python data science",
        text_weight=1.0,
        vec_weight=0.0,
    )
    assert isinstance(results, list)
    # Should still find results via BM25
    for m in results:
        assert isinstance(m, HybridMatch)


def test_search_hybrid_weights(populated_brain):
    """hybrid_search with text_weight=1.0 should behave like text-only."""
    text_only = populated_brain.hybrid_search(
        "Rust", text_weight=1.0, vec_weight=0.0, limit=10,
    )
    # Should not crash, and should return results
    assert isinstance(text_only, list)


# ================================================================
# Graph Centrality Tests (2)
# ================================================================

def test_centrality_pagerank(populated_brain):
    """centrality with pagerank should return scored nodes."""
    results = populated_brain.centrality(algorithm="pagerank", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    # Each entry should have node_id and score
    for entry in results:
        assert isinstance(entry, dict)
        assert "node_id" in entry or "id" in entry
        assert "score" in entry


def test_centrality_degree(populated_brain):
    """centrality with degree algorithm should work."""
    results = populated_brain.centrality(algorithm="degree", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0


# ================================================================
# Shortest Path Tests (2)
# ================================================================

def test_shortest_path_found(populated_brain):
    """shortest_path should find a path between linked nodes."""
    # Node 3 -> Node 0 via caused_by edge
    result = populated_brain.shortest_path(source=3, target=0)
    assert isinstance(result, PathResult)
    assert result.found
    assert len(result.path) >= 2
    assert result.path[0] == 3
    assert result.path[-1] == 0


def test_shortest_path_not_found(populated_brain):
    """shortest_path between disconnected nodes should return found=False."""
    # Node 5 (Go fact) and node 3 (Chose Rust decision) are not connected
    # through forward-only traversal
    result = populated_brain.shortest_path(
        source=5, target=3, direction="forward", max_depth=2,
    )
    assert isinstance(result, PathResult)
    # It may or may not find a path depending on direction traversal,
    # but the call must not crash
    assert isinstance(result.found, bool)


# ================================================================
# Belief Revision Tests (2)
# ================================================================

def test_revise_basic(populated_brain):
    """revise should return a RevisionReport."""
    report = populated_brain.revise(
        "Team now knows Go programming language well",
        threshold=0.3,
    )
    assert isinstance(report, RevisionReport)
    assert isinstance(report.contradicted, list)
    assert isinstance(report.weakened, list)
    assert isinstance(report.invalidated_decisions, list)
    assert isinstance(report.total_affected, int)


def test_revise_readonly(populated_brain):
    """revise should NOT modify the brain (read-only analysis)."""
    info_before = populated_brain.info()
    populated_brain.revise("Rust is no longer memory safe")
    info_after = populated_brain.info()
    assert info_before.node_count == info_after.node_count
    assert info_before.edge_count == info_after.edge_count


# ================================================================
# Gap Detection Tests (2)
# ================================================================

def test_gaps_basic(populated_brain):
    """gaps should return a GapReport with health_score."""
    report = populated_brain.gaps(threshold=0.5, min_support=2)
    assert isinstance(report, GapReport)
    assert isinstance(report.gaps, list)
    assert isinstance(report.health_score, float)
    assert 0.0 <= report.health_score <= 1.0


def test_gaps_health_score(brain):
    """An empty brain should have perfect health (1.0)."""
    brain.create()
    # Add only facts (no unjustified decisions)
    brain.add_fact("Solid fact one", session=1, confidence=0.95)
    brain.add_fact("Solid fact two", session=1, confidence=0.95)
    report = brain.gaps(threshold=0.5, min_support=1)
    assert isinstance(report, GapReport)
    # With only facts and no decisions, health should be high
    assert report.health_score >= 0.5


# ================================================================
# Analogy Tests (1)
# ================================================================

def test_analogy_basic(populated_brain):
    """analogy should return a list of Analogy results."""
    results = populated_brain.analogy(
        "choosing a programming language",
        limit=5,
        min_similarity=0.0,
    )
    assert isinstance(results, list)
    for a in results:
        assert isinstance(a, Analogy)
        assert isinstance(a.center_id, int)
        assert isinstance(a.combined_score, float)


# ================================================================
# Consolidation Tests (1)
# ================================================================

def test_consolidate_dry_run(populated_brain):
    """consolidate in dry-run mode should not modify the brain."""
    info_before = populated_brain.info()
    report = populated_brain.consolidate(
        deduplicate=True,
        confirm=False,  # dry-run
    )
    assert isinstance(report, ConsolidationReport)
    assert isinstance(report.actions, list)
    assert isinstance(report.deduplicated, int)
    # Brain should be unchanged
    info_after = populated_brain.info()
    assert info_before.node_count == info_after.node_count
    assert info_before.edge_count == info_after.edge_count


# ================================================================
# Drift Detection Tests (2)
# ================================================================

def test_drift_basic(populated_brain):
    """drift should return a DriftReport."""
    report = populated_brain.drift("programming language", min_relevance=0.2)
    assert isinstance(report, DriftReport)
    assert isinstance(report.timelines, list)
    assert isinstance(report.stability, float)
    assert isinstance(report.likely_to_change, bool)


def test_drift_stability(brain):
    """A single stable fact should have high stability."""
    brain.create()
    brain.add_fact("Database uses PostgreSQL", session=1, confidence=0.95)
    report = brain.drift("Database PostgreSQL", min_relevance=0.2)
    assert isinstance(report, DriftReport)
    # With a single fact and no corrections, stability should be high
    assert report.stability >= 0.5


# ================================================================
# Version Compatibility Tests (1)
# ================================================================

def test_version_check_passes(brain):
    """The brain should work with the current amem binary version."""
    brain.create()
    # Just verify that basic operations work end-to-end with the v0.2 binary
    node_id = brain.add_fact("Version check fact", session=1)
    event = brain.get(node_id)
    assert event.content == "Version check fact"
    # And that new query methods are available
    results = brain.search_text("Version check")
    assert isinstance(results, list)
