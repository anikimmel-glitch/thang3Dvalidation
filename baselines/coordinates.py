"""Coordinate changes derive from model inputs, never fitted predicted geometry."""
import numpy as np


def inverse_bbox(vertices, normalized_side):
    points=np.asarray(vertices)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or len(points)<2:
        raise ValueError("Invalid input coordinates")
    center=(points.min(axis=0)+points.max(axis=0))/2
    span=float(np.ptp(points,axis=0).max())
    if not np.isfinite(normalized_side) or normalized_side<=0 or span<=1e-10:
        raise ValueError("Degenerate input transform")
    return dict(scale=span/normalized_side,translation=center.tolist(),
                source="Inverse of model input bbox normalization; no output alignment")
