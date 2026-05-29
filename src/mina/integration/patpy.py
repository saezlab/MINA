from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

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
    Wrap a precomputed sample x feature matrix so it can be used with patpy
    downstream tools.

    This is useful when sample-level representations were computed outside
    patpy, for example using MINA, MOFA, PCA, or another latent-variable model.

    Parameters
    ----------
    representation
        DataFrame with samples in rows and latent variables/features in columns.
    metadata
        Optional sample-level metadata. Its index must contain the same sample
        IDs as ``representation.index``.
    metric
        Distance metric passed to ``scipy.spatial.distance.pdist``.
    seed
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
                    "`metadata` is missing samples from `representation.index`; "
                    f"for example: {missing[:5].tolist()}"
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
        No-op because the representation is already sample-level.
        """
        return None

    def calculate_distance_matrix(
        self,
        force: bool = False,
        dist: str | None = None,
    ) -> np.ndarray:
        """
        Calculate a sample x sample distance matrix from the precomputed
        representation.
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
        """

        metadata = self.metadata if metadata is None else metadata.copy()
        metadata.index = metadata.index.astype(str)

        return super().to_adata(
            metadata=metadata,
            *args,
            **kwargs,
        )
