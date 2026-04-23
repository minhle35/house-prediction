from app.transformer import ZeroToNan, TransformDate, PandasColumnTransform

import numpy as np
import pandas as pd


SCHEMA_SPECIFIC_BASE_STEPS = [
    # ===== Pre-processing: clean up data =====
    (
        "zero_to_nan",
        ZeroToNan(
            columns=[
                "num_bath",
                "num_bed",
                "property_size",
                "suburb_population",
                "suburb_median_income",
                "suburb_sqkm",
                "suburb_lat",
                "suburb_lng",
                "suburb_elevation",
                "property_inflation_index",
                "km_from_cbd",
                "suburb_median_house_price",
                "median_house_rent_per_week",
                "suburb_median_apartment_price",
                "median_apartment_rent_per_week",
                "avg_years_held",
                "time_to_cbd_public_transport_town_hall_st",
                "time_to_cbd_driving_town_hall_st",
                "traffic",
                "public_transport",
                "affordability_rental",
                "affordability_buying",
                "nature",
                "noise",
                "things_to_see_do",
                "family_friendliness",
                "pet_friendliness",
                "safety",
                "overall_rating",
            ]
        ),
    ),
    # ===== Feature engineering: adds derived features to DataFrame =====
    (
        "transform_date",
        TransformDate(
            target="date_sold",
            days_since="sell_days_since",
            days_since_relative_to_min="sell_days_since_min",
            year="sell_year",
            month="sell_month",
            season="sell_season",
        ),
    ),
    (
        "log_cols",
        PandasColumnTransform(
            func=np.log1p,
            columns=[
                "property_size",
                "km_from_cbd",
                "suburb_population",
                "suburb_median_house_price",
                "suburb_median_apartment_price",
            ],
            outputs=[
                "log_property_size",
                "log_km_from_cbd",
                "log_suburb_population",
                "log_suburb_median_house_price",
                "log_suburb_median_apartment_price",
            ],
        ),
    ),
    (
        "bin_transport_time_to",
        PandasColumnTransform(
            func=lambda x: pd.cut(
                x, bins=[-1, 30, 60, float("inf")], labels=["short", "medium", "long"]
            ).to_list(),
            columns=[
                "time_to_cbd_public_transport_town_hall_st",
                "time_to_cbd_driving_town_hall_st",
            ],
            outputs=[
                "time_to_cbd_public_bin",
                "time_to_cbd_townhall_driving_bin",
            ],
        ),
    ),
]
