"""Parameter sensitivity and robustness analysis."""

from quantproof.sensitivity.parameter_analysis import (
    SurfaceAnalysis,
    analyze_parameter_surface,
    render_ascii_surface,
    sensitivity_findings,
    surface_matrix,
)

__all__ = [
    "SurfaceAnalysis",
    "analyze_parameter_surface",
    "render_ascii_surface",
    "sensitivity_findings",
    "surface_matrix",
]
