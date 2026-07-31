"""Provide test cell graph signature functionality."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")
from benchmark.data.dataset import RegionData
from benchmark.features.cell_graph_signature import CellGraphSignatureBuilder
from benchmark.models.cell_graph_signature import CellGraphSignatureClassifier


def _region(region_id, offset=0.0):
    """Execute the region operation.
    
        Args:
            region_id (Any): Unique identifier of a tissue region.
            offset (Any): Numeric offset added when constructing synthetic test data.
    
        Returns:
            Any: The operation result.
    
    Args:
        region_id (Any): Unique identifier of a tissue region."""
    ids = pd.Index(range(6), name="cell_id")
    return RegionData(
        region_id=region_id,
        coordinates=pd.DataFrame(
            {"x": [0, 3, 30, 33, 60, 63], "y": np.zeros(6)}, index=ids
        ),
        expression=pd.DataFrame(
            {"m1": np.arange(1, 7) + offset, "m2": np.ones(6)}, index=ids
        ),
        cell_types=pd.DataFrame({"cell_type": ["a"] * 6}, index=ids),
    )


def test_builder_chunks_scales_and_connects():
    """Execute the test builder chunks scales and connects operation.

    Returns:
        Any: The operation result."""
    builder = CellGraphSignatureBuilder(graph_size=4, radius_um=20).fit([_region("a")])
    graphs = builder.extract_region(_region("a"))["graphs"]
    assert [graph.num_nodes for graph in graphs] == [4, 2]
    assert graphs[0].edge_index.shape == (2, 4)
    assert np.isclose(float(graphs[0].x[:, 0].max()), 4 / 6)


def test_classifier_region_probabilities():
    """Execute the test classifier region probabilities operation.

    Returns:
        Any: The operation result."""
    regions = [_region("a"), _region("b", 3), _region("c", 6), _region("d", 9)]
    builder = CellGraphSignatureBuilder(graph_size=4, radius_um=20).fit(regions)
    features = builder.transform(regions)
    target = pd.Series([0, 0, 1, 1], index=features.index)
    model = CellGraphSignatureClassifier(
        hidden_dim=8, epochs=1, patience=0, batch_size=4, device="cpu", seed=0
    ).fit(features, target)
    prediction = model.predict(features)
    assert prediction.shape == (4, 2)
    np.testing.assert_allclose(prediction.sum(axis=1), 1.0, atol=1e-6)
