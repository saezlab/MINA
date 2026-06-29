"""Compatibility helpers for optional patpy integrations."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

try:
    from patpy.tl import SampleRepresentationMethod
except ImportError as e:
    raise ImportError(
        "The patpy integration requires patpy. "
        "Install it with:\n\n"
        "    pip install 'mina[patpy]'\n\n"
        "or, in a uv-managed environment:\n\n"
        "    uv sync --extra patpy"
    ) from e


class PrecomputedSampleRepresentation(SampleRepresentationMethod):
    """
    Wrap a precomputed sample-by-feature matrix for patpy tools.

    This is useful when sample-level representations were computed outside
    patpy, for example using MINA, MOFA, PCA, or another latent-variable model.

    Parameters
    ----------
    representation : pandas.DataFrame
        DataFrame with samples in rows and latent variables/features in columns.
    metadata : pandas.DataFrame or None
        Optional sample-level metadata. Its index must contain the same sample
        IDs as ``representation.index``.
    metric : str
        Distance metric passed to ``scipy.spatial.distance.pdist``.
    seed : int
        Random seed used by patpy embedding methods.
    """

    DISTANCES_UNS_KEY = "X_precomputed_distances"

    def __init__(
        self,
        representation: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        metric: str = "euclidean",
        seed: int = 67,
    ):
        super().__init__(
            sample_key="sample_id",
            cell_group_key=None,
            layer=None,
            seed=seed,
        )

        if not isinstance(representation, pd.DataFrame):
            raise TypeError("`representation` must be a pandas DataFrame.")

        if representation.index.has_duplicates:
            raise ValueError("`representation.index` contains duplicated sample IDs.")

        if representation.columns.has_duplicates:
            raise ValueError("`representation.columns` contains duplicated feature names.")

        representation = representation.copy()
        representation.index = representation.index.astype(str)
        representation.columns = representation.columns.astype(str)

        if not np.all(np.isfinite(representation.to_numpy(dtype=float))):
            raise ValueError("`representation` must contain only finite numeric values.")

        self.metric = metric
        self.representation_df = representation
        self.sample_representation = representation
        self.samples = representation.index.to_numpy()
        self._distances: np.ndarray | None = None

        if metadata is not None:
            metadata = metadata.copy()
            metadata.index = metadata.index.astype(str)

            missing = representation.index.difference(metadata.index)
            if len(missing) > 0:
                raise ValueError(
                    f"`metadata` is missing samples from `representation.index`; for example: {missing[:5].tolist()}"
                )

            self.metadata = metadata.loc[representation.index].copy()
        else:
            self.metadata = pd.DataFrame(index=representation.index)

        obs = self.metadata.copy()
        obs["sample_id"] = obs.index.astype(str)

        self.adata = ad.AnnData(
            X=representation.to_numpy(dtype=float),
            obs=obs,
            var=pd.DataFrame(index=representation.columns),
        )

        self._fitted = True

    def prepare_anndata(self, adata=None):
        """
        Skip AnnData preparation.

        Parameters
        ----------
        adata : anndata.AnnData or None
            Ignored placeholder for compatibility with the patpy interface.

        Returns
        -------
        None
            The representation is already sample-level and no preparation is needed.
        """
        return None

    def calculate_distance_matrix(
        self,
        force: bool = False,
        dist: str | None = None,
    ) -> np.ndarray:
        """
        Calculate a sample-by-sample distance matrix.

        Parameters
        ----------
        force : bool
            If True, recompute distances even when cached distances exist.
        dist : str or None
            Distance metric passed to ``scipy.spatial.distance.pdist``. If None,
            ``self.metric`` is used.

        Returns
        -------
        distances : numpy.ndarray
            Square sample-by-sample distance matrix.
        """
        if self._distances is not None and not force:
            return self._distances

        metric = dist or self.metric

        self._distances = squareform(
            pdist(
                self.representation_df.to_numpy(dtype=float),
                metric=metric,
            )
        )

        self.adata.uns[self.DISTANCES_UNS_KEY] = self._distances

        return self._distances

    def to_adata(
        self,
        metadata: pd.DataFrame | None = None,
        *args,
        **kwargs,
    ) -> ad.AnnData:
        """
        Convert the precomputed sample representation to an AnnData object.

        Parameters
        ----------
        metadata : pandas.DataFrame or None
            Optional metadata to pass to the patpy conversion. If None, the
            metadata stored on the representation is used.
        *args : tuple
            Additional positional arguments passed to the parent method.
        **kwargs : dict
            Additional keyword arguments passed to the parent method.

        Returns
        -------
        adata : anndata.AnnData
            AnnData representation produced by the parent patpy method.
        """
        metadata = self.metadata if metadata is None else metadata.copy()
        metadata.index = metadata.index.astype(str)

        return super().to_adata(
            *args,
            metadata=metadata,
            **kwargs,
        )
