from dataclasses import dataclass


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

from app.transformer.base import BaseTransformer
from typing import Any


@dataclass
class CustomRegressionFeatures(BaseTransformer):
    """Custom feature engineering for regression.

    Parameters
    ----------
    sell_month_col: str, optional
        The name of column where the sell month is calculated. see `TransformDate`. If not set, will
        omit columns that depend on this.
    """

    sell_month_col: str | None = None

    def transform(self, X: pd.DataFrame) -> Any:
        if self.sell_month_col:
            X["month_sin"] = np.sin(2 * np.pi * X[self.sell_month_col] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X[self.sell_month_col] / 12)
        X["bed_bath_combo"] = X["num_bed"] * X["num_bath"]
        X["size_income_ratio"] = X["log_property_size"] / (
            X["suburb_median_income"] + 1
        )
        X["median_price_per_km"] = (
            X["log_suburb_median_house_price"] / X["log_km_from_cbd"]
        )
        X["price_distance_ratio"] = (
            X["log_suburb_median_house_price"] / X["log_km_from_cbd"]
        )
        X["population_density"] = X["log_suburb_population"] / X["suburb_sqkm"]
        X["bedroom_density"] = X["num_bed"] / X["log_property_size"]
        X["size_per_bedroom"] = X["property_size"] / X["num_bed"].clip(lower=1)
        X["area_price_per_sqm"] = X["suburb_median_house_price"] / X[
            "property_size"
        ].clip(lower=1)
        X["rent_yield"] = (
            X["median_house_rent_per_week"] * 52 / (X["suburb_median_house_price"] + 1)
        )
        X["income_size_combo"] = X["suburb_median_income"] * X["log_property_size"]
        X["total_rooms"] = X["num_bed"] + X["num_bath"]
        X["total_spaces"] = X["total_rooms"] + X["num_parking"]

        clustering_features = [
            "suburb_lat",
            "suburb_lng",
            "log_suburb_median_house_price",
            "log_km_from_cbd",
        ]
        clustering_df = X[clustering_features].dropna()
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(clustering_df)
        kmeans = KMeans(n_clusters=20, random_state=42)
        clusters = kmeans.fit_predict(scaled)
        X.loc[clustering_df.index, "postcode_cluster"] = clusters
        X["postcode_cluster"] = X["postcode_cluster"].fillna(-1).astype(int).astype(str)
        X["region_postcode_cluster"] = (
            X["region"].astype(str) + "_" + X["postcode_cluster"]
        )
        return X


@dataclass
class PropertyTypeFeatures(BaseTransformer):
    """Adds property-type specific features to help classification."""

    def transform(self, X: pd.DataFrame) -> Any:
        # Duplex-specific features
        X["bath_bed_ratio"] = X["num_bath"] / X["num_bed"].clip(lower=1)
        X["duplex_bath_ratio"] = (X["bath_bed_ratio"] > 0.6).astype(int)

        X["size_per_bedroom"] = (
            X["property_size"] / X["num_bed"].clip(lower=1)
            if "size_per_bedroom" not in X.columns
            else X["size_per_bedroom"]
        )
        X["duplex_size_bedroom"] = (X["size_per_bedroom"] < 133).astype(int)

        X["total_rooms"] = (
            X["num_bed"] + X["num_bath"]
            if "total_rooms" not in X.columns
            else X["total_rooms"]
        )
        X["duplex_total_rooms"] = (X["total_rooms"] > 6).astype(int)

        X["parking_bed_ratio"] = X["num_parking"] / X["num_bed"].clip(lower=1)
        X["duplex_parking_ratio"] = (X["parking_bed_ratio"] < 0.5).astype(int)

        if "price_distance_ratio" in X.columns:
            X["duplex_price_dist"] = (X["price_distance_ratio"] < 115000).astype(int)
        else:
            X["duplex_price_dist"] = 0

        # Combined score for Duplex likelihood
        X["likely_duplex"] = (
            X["duplex_price_dist"] * 0.25
            + X.get("duplex_public_housing", 0) * 0.15
            + X["duplex_size_bedroom"] * 0.20
            + X["duplex_bath_ratio"] * 0.15
            + X["duplex_total_rooms"] * 0.15
            + X["duplex_parking_ratio"] * 0.10
        )

        # Add features for other rare property types here...

        return X
