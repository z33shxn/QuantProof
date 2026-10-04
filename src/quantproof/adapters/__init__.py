"""Adapters that convert engine-specific outputs into :class:`ResearchArtifacts`."""

from quantproof.adapters.base import Adapter, ResearchArtifacts
from quantproof.adapters.generic import GenericResultsAdapter
from quantproof.adapters.pandas import PandasAdapter

__all__ = ["Adapter", "GenericResultsAdapter", "PandasAdapter", "ResearchArtifacts"]
