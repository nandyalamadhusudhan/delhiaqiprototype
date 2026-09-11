import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. LOAD DATASETS
# =========================================================

fire = pd.read_csv(
    "firedata.csv"
)

weather = pd.read_csv(
    "delhi-weather-aqi-2025.csv"
)

print("\nDatasets loaded successfully.")


# =========================================================
# 2. CONVERT FIRE ACQUISITION DATE
# =========================================================

# Fire dataset example:
# 2025-01-01

fire["acq_date"] = pd.to_datetime(
    fire["acq_date"],
    format="mixed",
    errors="coerce"
)


# =========================================================
# 3. REMOVE INVALID FIRE DATES
# =========================================================

fire = fire.dropna(
    subset=["acq_date"]
).reset_index(drop=True)


# =========================================================
# 4. CONVERT ACQUISITION TIME
# =========================================================

fire["acq_time"] = (
    pd.to_numeric(
        fire["acq_time"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# =========================================================
# 5. EXTRACT FIRE HOUR
# =========================================================

fire["fire_hour"] = (
    fire["acq_time"] // 100
)


# =========================================================
# 6. EXTRACT FIRE MINUTE
# =========================================================

fire["fire_minute"] = (
    fire["acq_time"] % 100
)


# =========================================================
# 7. CREATE UTC TIMESTAMP
# =========================================================

fire["timestamp_utc"] = (
    fire["acq_date"]
    + pd.to_timedelta(
        fire["fire_hour"],
        unit="h"
    )
    + pd.to_timedelta(
        fire["fire_minute"],
        unit="m"
    )
)


# =========================================================
# 8. CONVERT UTC TO IST
# =========================================================

fire["timestamp_ist"] = (
    fire["timestamp_utc"]
    + pd.Timedelta(
        hours=5,
        minutes=30
    )
)


# =========================================================
# 9. CONVERT FIRE DATA TO HOURLY TIMESTAMP
# =========================================================

fire["timestamp"] = (
    fire["timestamp_ist"]
    .dt.floor("h")
)


# =========================================================
# 10. CHECK FIRE TIMESTAMP
# =========================================================

print("\n==========================================")
print("FIRE TIMESTAMP CHECK")
print("==========================================")

print(
    fire[
        [
            "acq_date",
            "acq_time",
            "timestamp_utc",
            "timestamp_ist",
            "timestamp"
        ]
    ].head(10)
)


# =========================================================
# 11. FIRE COUNT
# =========================================================

fire_hourly = (
    fire
    .groupby("timestamp")
    .agg(
        fire_count=(
            "latitude",
            "count"
        )
    )
    .reset_index()
)


# =========================================================
# 12. FIRE FRP
# =========================================================

fire_frp = (
    fire
    .groupby("timestamp")
    .agg(
        fire_frp_sum=(
            "frp",
            "sum"
        )
    )
    .reset_index()
)


# =========================================================
# 13. FIRE BRIGHTNESS
# =========================================================

fire_brightness = (
    fire
    .groupby("timestamp")
    .agg(
        fire_brightness_sum=(
            "brightness",
            "sum"
        )
    )
    .reset_index()
)


# =========================================================
# 14. MERGE BASIC FIRE FEATURES
# =========================================================

fire_hourly = fire_hourly.merge(
    fire_frp,
    on="timestamp",
    how="left"
)

fire_hourly = fire_hourly.merge(
    fire_brightness,
    on="timestamp",
    how="left"
)


# =========================================================
# 15. REGION CLASSIFICATION
# =========================================================

def classify_region(latitude, longitude):

    # Delhi NCR
    if (
        28.30 <= latitude <= 29.00
        and
        76.80 <= longitude <= 77.80
    ):
        return "delhi_ncr"

    # Haryana
    elif (
        27.50 <= latitude <= 30.90
        and
        74.80 <= longitude <= 77.60
    ):
        return "haryana"

    # Punjab
    elif (
        29.50 <= latitude <= 32.50
        and
        73.80 <= longitude <= 76.90
    ):
        return "punjab"

    # Uttar Pradesh
    elif (
        24.00 <= latitude <= 30.50
        and
        77.00 <= longitude <= 84.50
    ):
        return "uttar_pradesh"

    # Uttarakhand
    elif (
        28.70 <= latitude <= 31.50
        and
        77.50 <= longitude <= 81.00
    ):
        return "uttarakhand"

    else:
        return "other"


fire["region"] = fire.apply(
    lambda row: classify_region(
        row["latitude"],
        row["longitude"]
    ),
    axis=1
)


# =========================================================
# 16. REGION FIRE COUNTS
# =========================================================

region_fire_hourly = (
    fire
    .pivot_table(
        index="timestamp",
        columns="region",
        values="latitude",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)


# =========================================================
# 17. RENAME REGION COLUMNS
# =========================================================

region_fire_hourly = region_fire_hourly.rename(
    columns={
        "punjab":
            "punjab_fire_count",

        "haryana":
            "haryana_fire_count",

        "uttar_pradesh":
            "uttar_pradesh_fire_count",

        "uttarakhand":
            "uttarakhand_fire_count",

        "delhi_ncr":
            "delhi_ncr_fire_count"
    }
)


# =========================================================
# 18. ENSURE REGION COLUMNS EXIST
# =========================================================

region_columns = [
    "punjab_fire_count",
    "haryana_fire_count",
    "uttar_pradesh_fire_count",
    "uttarakhand_fire_count",
    "delhi_ncr_fire_count"
]


for column in region_columns:

    if column not in region_fire_hourly.columns:

        region_fire_hourly[column] = 0


# =========================================================
# 19. SELECT REGION COLUMNS
# =========================================================

region_fire_hourly = region_fire_hourly[
    [
        "timestamp",

        "punjab_fire_count",
        "haryana_fire_count",
        "uttar_pradesh_fire_count",
        "uttarakhand_fire_count",
        "delhi_ncr_fire_count"
    ]
]


# =========================================================
# 20. DELHI REFERENCE LOCATION
# =========================================================

DELHI_LAT = 28.6469
DELHI_LON = 77.3160


# =========================================================
# 21. CALCULATE FIRE BEARING
# =========================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    difference_lon = np.radians(
        lon2 - lon1
    )

    x = (
        np.sin(difference_lon)
        *
        np.cos(lat2)
    )

    y = (
        np.cos(lat1)
        *
        np.sin(lat2)
        -
        np.sin(lat1)
        *
        np.cos(lat2)
        *
        np.cos(difference_lon)
    )

    bearing = np.degrees(
        np.arctan2(x, y)
    )

    bearing = (
        bearing + 360
    ) % 360

    return bearing


# =========================================================
# 22. FIRE BEARING
# =========================================================

fire["fire_bearing"] = fire.apply(
    lambda row: calculate_bearing(
        DELHI_LAT,
        DELHI_LON,
        row["latitude"],
        row["longitude"]
    ),
    axis=1
)


# =========================================================
# 23. WEATHER TIMESTAMP
# =========================================================

weather["timestamp"] = pd.to_datetime(
    weather["date_ist"].astype(str)
    + " "
    + weather["time_ist"].astype(str),
    dayfirst=True,
    errors="coerce"
)


# =========================================================
# 24. REMOVE INVALID WEATHER TIMESTAMPS
# =========================================================

weather = weather.dropna(
    subset=["timestamp"]
).reset_index(drop=True)


# =========================================================
# 25. WEATHER WIND DATA
# =========================================================

wind_column = (
    "Wind (From) Direction (Degrees)"
)


wind_data = weather[
    [
        "timestamp",
        wind_column
    ]
].copy()


wind_data = wind_data.rename(
    columns={
        wind_column:
        "wind_direction"
    }
)


# =========================================================
# 26. MERGE FIRE WITH WIND
# =========================================================

fire_with_wind = fire.merge(
    wind_data,
    on="timestamp",
    how="left"
)


# =========================================================
# 27. CALCULATE WIND ANGLE
# =========================================================

fire_with_wind["angle_difference"] = (
    abs(
        fire_with_wind["fire_bearing"]
        -
        fire_with_wind["wind_direction"]
    )
)


fire_with_wind["angle_difference"] = (
    fire_with_wind["angle_difference"]
    .apply(
        lambda x:
        min(
            x,
            360 - x
        )
        if pd.notna(x)
        else np.nan
    )
)


# =========================================================
# 28. IDENTIFY UPWIND FIRES
# =========================================================

fire_with_wind["is_upwind"] = (
    fire_with_wind["angle_difference"]
    <= 45
)


# =========================================================
# 29. STUBBLE FIRE REGIONS
# =========================================================

stubble_regions = [
    "punjab",
    "haryana",
    "uttar_pradesh"
]


fire_with_wind["is_upwind_stubble"] = (
    fire_with_wind["is_upwind"]
    &
    fire_with_wind["region"].isin(
        stubble_regions
    )
)


# =========================================================
# 30. HOURLY UPWIND STUBBLE FIRES
# =========================================================

upwind_fire_hourly = (
    fire_with_wind[
        fire_with_wind["is_upwind_stubble"]
    ]
    .groupby("timestamp")
    .size()
    .reset_index(
        name="upwind_stubble_fire_count"
    )
)


# =========================================================
# 31. CREATE FINAL DATASET
# =========================================================

final_data = weather.merge(
    fire_hourly,
    on="timestamp",
    how="left"
)


# =========================================================
# 32. MERGE REGION FIRE FEATURES
# =========================================================

final_data = final_data.merge(
    region_fire_hourly,
    on="timestamp",
    how="left"
)


# =========================================================
# 33. MERGE UPWIND FIRE FEATURE
# =========================================================

final_data = final_data.merge(
    upwind_fire_hourly,
    on="timestamp",
    how="left"
)


# =========================================================
# 34. FILL FIRE VALUES
# =========================================================

fire_columns = [
    "fire_count",
    "fire_frp_sum",
    "fire_brightness_sum",

    "punjab_fire_count",
    "haryana_fire_count",
    "uttar_pradesh_fire_count",
    "uttarakhand_fire_count",
    "delhi_ncr_fire_count",

    "upwind_stubble_fire_count"
]


for column in fire_columns:

    if column in final_data.columns:

        final_data[column] = (
            final_data[column]
            .fillna(0)
        )


# =========================================================
# 35. CHECK TEMPERATURE
# =========================================================

print("\n==========================================")
print("TEMPERATURE CHECK")
print("==========================================")

print(
    final_data[
        [
            "timestamp",
            "temp_c"
        ]
    ].head(10)
)


# =========================================================
# 36. SAVE FINAL DATASET
# =========================================================

final_data.to_csv(
    "delhi_final_dataset.csv",
    index=False
)


print("\nFinal dataset saved:")
print("delhi_final_dataset.csv")


# =========================================================
# 37. SORT FINAL DATA
# =========================================================

final_data = final_data.sort_values(
    "timestamp"
).reset_index(drop=True)


# =========================================================
# 38. CREATE TIME FEATURES
# =========================================================

final_data["hour"] = (
    final_data["timestamp"].dt.hour
)

final_data["day_of_week"] = (
    final_data["timestamp"].dt.dayofweek
)

final_data["month"] = (
    final_data["timestamp"].dt.month
)


# =========================================================
# 39. CREATE SEASON FEATURES
# =========================================================

final_data["is_winter"] = (
    final_data["month"]
    .isin([11, 12, 1, 2])
    .astype(int)
)


final_data["is_stubble_season"] = (
    final_data["month"]
    .isin([10, 11])
    .astype(int)
)


# =========================================================
# 40. CREATE HISTORICAL AQI FEATURES
# =========================================================

final_data["aqi_lag1"] = (
    final_data["aqi_index"].shift(1)
)

final_data["aqi_lag6"] = (
    final_data["aqi_index"].shift(6)
)

final_data["aqi_lag12"] = (
    final_data["aqi_index"].shift(12)
)

final_data["aqi_lag24"] = (
    final_data["aqi_index"].shift(24)
)

final_data["aqi_lag48"] = (
    final_data["aqi_index"].shift(48)
)

final_data["aqi_lag72"] = (
    final_data["aqi_index"].shift(72)
)


# =========================================================
# 41. CREATE PM2.5 HISTORICAL FEATURES
# =========================================================

final_data["pm25_lag1"] = (
    final_data["pm2_5"].shift(1)
)

final_data["pm25_lag6"] = (
    final_data["pm2_5"].shift(6)
)

final_data["pm25_lag12"] = (
    final_data["pm2_5"].shift(12)
)

final_data["pm25_lag24"] = (
    final_data["pm2_5"].shift(24)
)

final_data["pm25_lag48"] = (
    final_data["pm2_5"].shift(48)
)

final_data["pm25_lag72"] = (
    final_data["pm2_5"].shift(72)
)


# =========================================================
# 42. CREATE PM10 HISTORICAL FEATURES
# =========================================================

final_data["pm10_lag1"] = (
    final_data["pm10"].shift(1)
)

final_data["pm10_lag6"] = (
    final_data["pm10"].shift(6)
)

final_data["pm10_lag12"] = (
    final_data["pm10"].shift(12)
)

final_data["pm10_lag24"] = (
    final_data["pm10"].shift(24)
)

final_data["pm10_lag48"] = (
    final_data["pm10"].shift(48)
)

final_data["pm10_lag72"] = (
    final_data["pm10"].shift(72)
)


# =========================================================
# 43. CREATE ROLLING AQI FEATURES
# =========================================================

final_data["aqi_rolling_6"] = (
    final_data["aqi_index"]
    .rolling(6)
    .mean()
)

final_data["aqi_rolling_12"] = (
    final_data["aqi_index"]
    .rolling(12)
    .mean()
)

final_data["aqi_rolling_24"] = (
    final_data["aqi_index"]
    .rolling(24)
    .mean()
)

final_data["aqi_rolling_48"] = (
    final_data["aqi_index"]
    .rolling(48)
    .mean()
)

final_data["aqi_rolling_72"] = (
    final_data["aqi_index"]
    .rolling(72)
    .mean()
)


# =========================================================
# 44. CREATE ROLLING PM2.5 FEATURES
# =========================================================

final_data["pm25_rolling_6"] = (
    final_data["pm2_5"]
    .rolling(6)
    .mean()
)

final_data["pm25_rolling_24"] = (
    final_data["pm2_5"]
    .rolling(24)
    .mean()
)

final_data["pm25_rolling_72"] = (
    final_data["pm2_5"]
    .rolling(72)
    .mean()
)


# =========================================================
# 45. CREATE FUTURE AQI TARGETS
# =========================================================

final_data["aqi_1h"] = (
    final_data["aqi_index"].shift(-1)
)

final_data["aqi_24h"] = (
    final_data["aqi_index"].shift(-24)
)

final_data["aqi_72h"] = (
    final_data["aqi_index"].shift(-72)
)


# =========================================================
# 46. SAVE DATASET WITH ML FEATURES
# =========================================================

final_data.to_csv(
    "delhi_final_ml_dataset.csv",
    index=False
)


print("\n==========================================")
print("FINAL ML DATASET CREATED")
print("==========================================")

print(
    "delhi_final_ml_dataset.csv"
)


# =========================================================
# 47. DEFINE AVAILABLE FEATURES
# =========================================================

features = [

    # -------------------------
    # TIME
    # -------------------------

    "hour",
    "day_of_week",
    "month",

    # -------------------------
    # SEASON
    # -------------------------

    "is_winter",
    "is_stubble_season",

    # -------------------------
    # WEATHER
    # -------------------------

    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",

    # -------------------------
    # CURRENT POLLUTION
    # -------------------------

    "aqi_index",
    "pm2_5",
    "pm10",
    "co",
    "no2",

    # -------------------------
    # FIRE
    # -------------------------

    "fire_count",
    "fire_frp_sum",
    "fire_brightness_sum",

    # -------------------------
    # REGION FIRE
    # -------------------------

    "punjab_fire_count",
    "haryana_fire_count",
    "uttar_pradesh_fire_count",
    "uttarakhand_fire_count",
    "delhi_ncr_fire_count",

    # -------------------------
    # UPWIND FIRE
    # -------------------------

    "upwind_stubble_fire_count",

    # -------------------------
    # AQI HISTORY
    # -------------------------

    "aqi_lag1",
    "aqi_lag6",
    "aqi_lag12",
    "aqi_lag24",
    "aqi_lag48",
    "aqi_lag72",

    # -------------------------
    # PM2.5 HISTORY
    # -------------------------

    "pm25_lag1",
    "pm25_lag6",
    "pm25_lag12",
    "pm25_lag24",
    "pm25_lag48",
    "pm25_lag72",

    # -------------------------
    # PM10 HISTORY
    # -------------------------

    "pm10_lag1",
    "pm10_lag6",
    "pm10_lag12",
    "pm10_lag24",
    "pm10_lag48",
    "pm10_lag72",

    # -------------------------
    # ROLLING AQI
    # -------------------------

    "aqi_rolling_6",
    "aqi_rolling_12",
    "aqi_rolling_24",
    "aqi_rolling_48",
    "aqi_rolling_72",

    # -------------------------
    # ROLLING PM2.5
    # -------------------------

    "pm25_rolling_6",
    "pm25_rolling_24",
    "pm25_rolling_72"
]


# =========================================================
# 48. CHECK FEATURES
# =========================================================

missing_features = [
    column
    for column in features
    if column not in final_data.columns
]


print("\n==========================================")
print("FEATURE CHECK")
print("==========================================")

print(
    "Missing features:",
    missing_features
)


# =========================================================
# 49. STOP IF FEATURES ARE MISSING
# =========================================================

if len(missing_features) > 0:

    print(
        "\nERROR: Some required features are missing."
    )

    print(
        "Available columns:"
    )

    print(
        final_data.columns.tolist()
    )

    raise SystemExit


# =========================================================
# 50. REMOVE MISSING VALUES
# =========================================================

data = final_data.dropna(
    subset=features + [
        "aqi_1h",
        "aqi_24h",
        "aqi_72h"
    ]
).reset_index(drop=True)


# =========================================================
# 51. TRAIN / TEST SPLIT
# =========================================================

split = int(
    len(data) * 0.8
)


train = data.iloc[
    :split
]


test = data.iloc[
    split:
]


# =========================================================
# 52. CREATE X TRAIN / X TEST
# =========================================================

X_train = train[
    features
]

X_test = test[
    features
]


# =========================================================
# 53. RANDOM FOREST FUNCTION
# =========================================================

def train_model(target):

    print("\n==========================================")
    print("Training:", target)
    print("==========================================")

    y_train = train[
        target
    ]

    y_test = test[
        target
    ]


    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )


    # Train
    model.fit(
        X_train,
        y_train
    )


    # Prediction
    prediction = model.predict(
        X_test
    )


    # MAE
    mae = mean_absolute_error(
        y_test,
        prediction
    )


    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    # R2
    r2 = r2_score(
        y_test,
        prediction
    )


    print(
        "MAE :",
        mae
    )

    print(
        "RMSE:",
        rmse
    )

    print(
        "R²  :",
        r2
    )


    return (
        model,
        prediction,
        mae,
        rmse,
        r2
    )


# =========================================================
# 54. TRAIN 1-HOUR MODEL
# =========================================================

model_1h, pred_1h, mae_1h, rmse_1h, r2_1h = (
    train_model(
        "aqi_1h"
    )
)


# =========================================================
# 55. TRAIN 24-HOUR MODEL
# =========================================================

model_24h, pred_24h, mae_24h, rmse_24h, r2_24h = (
    train_model(
        "aqi_24h"
    )
)


# =========================================================
# 56. TRAIN 72-HOUR MODEL
# =========================================================

model_72h, pred_72h, mae_72h, rmse_72h, r2_72h = (
    train_model(
        "aqi_72h"
    )
)


# =========================================================
# 57. MODEL PERFORMANCE SUMMARY
# =========================================================

print("\n")
print("==========================================")
print("          MODEL PERFORMANCE")
print("==========================================")


print("\nNext 1 Hour")

print(
    "MAE :",
    mae_1h
)

print(
    "RMSE:",
    rmse_1h
)

print(
    "R²  :",
    r2_1h
)


print("\nNext 24 Hours")

print(
    "MAE :",
    mae_24h
)

print(
    "RMSE:",
    rmse_24h
)

print(
    "R²  :",
    r2_24h
)


print("\nNext 72 Hours")

print(
    "MAE :",
    mae_72h
)

print(
    "RMSE:",
    rmse_72h
)

print(
    "R²  :",
    r2_72h
)


# =========================================================
# 58. CREATE PREDICTION COMPARISON
# =========================================================

comparison = test[
    [
        "timestamp",
        "aqi_index",
        "aqi_1h",
        "aqi_24h",
        "aqi_72h"
    ]
].copy()


comparison[
    "predicted_aqi_1h"
] = pred_1h


comparison[
    "predicted_aqi_24h"
] = pred_24h


comparison[
    "predicted_aqi_72h"
] = pred_72h


# =========================================================
# 59. CALCULATE ERRORS
# =========================================================

comparison[
    "error_1h"
] = (
    comparison["aqi_1h"]
    -
    comparison["predicted_aqi_1h"]
).abs()


comparison[
    "error_24h"
] = (
    comparison["aqi_24h"]
    -
    comparison["predicted_aqi_24h"]
).abs()


comparison[
    "error_72h"
] = (
    comparison["aqi_72h"]
    -
    comparison["predicted_aqi_72h"]
).abs()


# =========================================================
# 60. PRINT PREDICTIONS
# =========================================================

print("\n==========================================")
print("AQI PREDICTION COMPARISON")
print("==========================================")

print(
    comparison.head(20)
)


# =========================================================
# 61. SAVE PREDICTIONS
# =========================================================

comparison.to_csv(
    "aqi_prediction_results.csv",
    index=False
)


print(
    "\nPrediction results saved:"
)

print(
    "aqi_prediction_results.csv"
)


# =========================================================
# 62. SHAP FUNCTION
# =========================================================

def get_prediction_reason(
    model,
    X_row,
    top_n=5
):

    print(
        "\nCalculating SHAP explanation..."
    )


    explainer = shap.TreeExplainer(
        model
    )


    shap_values = (
        explainer.shap_values(
            X_row
        )
    )


    values = shap_values[0]


    explanation = pd.DataFrame({

        "feature":
            X_row.columns,

        "shap_value":
            values,

        "feature_value":
            X_row.iloc[0].values

    })


    explanation[
        "importance"
    ] = (
        explanation[
            "shap_value"
        ].abs()
    )


    explanation = (
        explanation
        .sort_values(
            "importance",
            ascending=False
        )
        .head(top_n)
    )


    explanation[
        "effect"
    ] = explanation[
        "shap_value"
    ].apply(

        lambda x:
        "Increases AQI"
        if x > 0
        else
        "Decreases AQI"

    )


    return explanation


# =========================================================
# 63. SELECT ONE TEST ROW
# =========================================================

row_number = 100


if row_number >= len(X_test):

    row_number = 0


current_row = X_test.iloc[
    [row_number]
]


# =========================================================
# 64. SHAP - 1 HOUR
# =========================================================

reason_1h = get_prediction_reason(
    model_1h,
    current_row,
    top_n=5
)


print("\n==========================================")
print("1-HOUR AQI REASONS")
print("==========================================")

print(
    reason_1h
)


# =========================================================
# 65. SHAP - 24 HOURS
# =========================================================

reason_24h = get_prediction_reason(
    model_24h,
    current_row,
    top_n=5
)


print("\n==========================================")
print("24-HOUR AQI REASONS")
print("==========================================")

print(
    reason_24h
)


# =========================================================
# 66. SHAP - 72 HOURS
# =========================================================

reason_72h = get_prediction_reason(
    model_72h,
    current_row,
    top_n=5
)


print("\n==========================================")
print("72-HOUR AQI REASONS")
print("==========================================")

print(
    reason_72h
)


# =========================================================
# 67. SAVE SHAP RESULTS
# =========================================================

reason_1h.to_csv(
    "shap_reasons_1h.csv",
    index=False
)

reason_24h.to_csv(
    "shap_reasons_24h.csv",
    index=False
)

reason_72h.to_csv(
    "shap_reasons_72h.csv",
    index=False
)


print("\n==========================================")
print("SHAP RESULTS SAVED")
print("==========================================")

print(
    "shap_reasons_1h.csv"
)

print(
    "shap_reasons_24h.csv"
)

print(
    "shap_reasons_72h.csv"
)