from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TripFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build features available before the trip is completed."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: ANN001
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        rename_map = {
            "pickup_datetime": "tpep_pickup_datetime",
            "pickup_location_id": "PULocationID",
            "dropoff_location_id": "DOLocationID",
            "rate_code_id": "RatecodeID",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        pickup = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df["pickup_hour"] = pickup.dt.hour
        df["pickup_dayofweek"] = pickup.dt.dayofweek
        df["pickup_month"] = pickup.dt.month
        df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
        df["pickup_dropoff_same_zone"] = (df["PULocationID"] == df["DOLocationID"]).astype(int)

        for column in ["PULocationID", "DOLocationID", "RatecodeID", "payment_type"]:
            df[column] = df[column].astype("string")

        return df
