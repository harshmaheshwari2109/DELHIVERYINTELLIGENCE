import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Cleans raw shipment data and returns cleaned df and medians dictionary."""
    df = raw.copy()

    # i. Drop exact duplicate rows
    df = df.drop_duplicates()

    # ii. Clean service_type: strip spaces and title case
    if "service_type" in df.columns:
        df["service_type"] = (
            df["service_type"].astype(str).str.strip().str.title()
        )

    # iii. Replace missing payment mode with "Unknown"
    if "payment_mode" in df.columns:
        df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # iv. Clean declared_value: remove rupee symbol, commas, convert to numeric
    if "declared_value" in df.columns:
        df["declared_value_inr"] = (
            df["declared_value"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["declared_value_inr"] = pd.to_numeric(
            df["declared_value_inr"], errors="coerce"
        )
        df = df.drop(columns=["declared_value"])

    # v. Convert dispatch_time to datetime
    if "dispatch_time" in df.columns:
        df["dispatch_time"] = pd.to_datetime(
            df["dispatch_time"], errors="coerce"
        )

    # vi. Handle scanner-fault sentinel (weights > 40 set to NaN)
    if "parcel_weight_kg" in df.columns:
        df.loc[df["parcel_weight_kg"] > 40, "parcel_weight_kg"] = np.nan

    # vii. Compute or apply medians for imputation
    if medians is None:
        medians = {
            "distance_km": df["distance_km"].median(),
            "parcel_weight_kg": df["parcel_weight_kg"].median(),
            "declared_value_inr": df["declared_value_inr"].median(),
        }

    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df, medians


def build_features(clean, hubs):
    """Merges hub data and constructs the 17 features required for modeling."""
    df = clean.merge(hubs, on="hub_id", how="left")

    # Promised hours mapping
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)

    # Payment mode indicator (1 if COD, else 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # Weekend indicator (Saturday = 5, Sunday = 6)
    df["is_weekend"] = df["dispatch_time"].dt.dayofweek.ge(5).astype(int)

    # Dispatch hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # Log declared value
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    # Ensure all 17 features exist and order them correctly
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df