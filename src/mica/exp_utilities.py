"""Per-member utility matrices."""
from __future__ import annotations

import numpy as np
import pandas as pd


def category_matrix(pois: pd.DataFrame, min_support: int = 5) -> pd.DataFrame:
    cats = pois.set_index("business_id")["categories"].str.get_dummies(sep=", ")
    keep = cats.columns[cats.sum() >= min_support]
    if len(keep) == 0:
        raise ValueError("no category survives the support filter")
    return cats[keep]


def _row_normalise(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norm == 0.0, 1.0, norm)


def content_profiles(
    train: pd.DataFrame, cats: pd.DataFrame, users: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_vectors = pd.DataFrame(
        _row_normalise(cats.to_numpy(dtype=float)), index=cats.index, columns=cats.columns
    )
    joined = train.loc[train["user_id"].isin(users)].join(cats, on="business_id", how="inner")
    if joined.empty:
        raise ValueError("no training reviews of catalogued POIs for these users")
    profile = joined.groupby("user_id")[list(cats.columns)].mean()
    user_vectors = pd.DataFrame(
        _row_normalise(profile.to_numpy(dtype=float)), index=profile.index, columns=cats.columns
    )
    return item_vectors, user_vectors


def to_utility(stars) -> np.ndarray:
    return np.clip((np.asarray(stars, dtype=float) - 1.0) / 4.0, 0.0, 1.0)


def build_utility_matrix(
    member_ids,
    pois: pd.DataFrame,
    global_mean: float,
    user_bias: pd.Series,
    item_bias: pd.Series,
    item_vectors: pd.DataFrame,
    user_vectors: pd.DataFrame,
    blend: float = 0.5,
) -> np.ndarray:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must lie in [0,1]")

    poi_ids = pois["business_id"].to_numpy()
    item_term = pois["business_id"].map(item_bias).fillna(0.0).to_numpy(dtype=float)
    known = np.array([pid in item_vectors.index for pid in poi_ids])
    content_items = np.zeros((len(poi_ids), item_vectors.shape[1]))
    content_items[known] = item_vectors.loc[poi_ids[known]].to_numpy(dtype=float)

    matrix = np.zeros((len(member_ids), len(poi_ids)))
    for i, member in enumerate(member_ids):
        bias_component = to_utility(
            np.clip(global_mean + float(user_bias.get(member, 0.0)) + item_term, 1.0, 5.0)
        )
        if member in user_vectors.index:
            content = content_items @ user_vectors.loc[member].to_numpy(dtype=float)
        else:
            content = np.zeros(len(poi_ids))
        matrix[i] = (1.0 - blend) * bias_component + blend * np.clip(content, 0.0, 1.0)
    return np.clip(matrix, 0.0, 1.0)
