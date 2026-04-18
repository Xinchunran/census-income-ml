from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from model.baseline.preprocess.features import CensusPreprocessor, build_preprocessor


@dataclass
class SegmenterArtifact:
    preprocessor: CensusPreprocessor
    kmeans: MiniBatchKMeans


def fit_segmenter(df: pd.DataFrame, n_clusters: int = 6, seed: int = 42) -> SegmenterArtifact:
    preprocessor = build_preprocessor(df)
    X = preprocessor.transform(df)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=4096, n_init=10)
    kmeans.fit(X)
    return SegmenterArtifact(preprocessor=preprocessor, kmeans=kmeans)


def assign_segments(segmenter: SegmenterArtifact, df: pd.DataFrame) -> np.ndarray:
    X = segmenter.preprocessor.transform(df)
    return segmenter.kmeans.predict(X).astype(int)
