
import os
import re
import json
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ----------------------------
# Utility / setup
# ----------------------------
def ensure_dirs():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)


def normalize_column_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def load_csv(input_path=None):
    raw_dir = Path("data/raw")

    if input_path:
        csv_path = Path(input_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
    else:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Folder does not exist: {raw_dir.resolve()}")

        csv_files = list(raw_dir.glob("*.csv"))
        if not csv_files:
            all_files = [f.name for f in raw_dir.iterdir()]
            raise FileNotFoundError(
                f"No CSV found inside {raw_dir.resolve()}\n"
                f"Files currently present: {all_files}"
            )
        csv_path = csv_files[0]

    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    return df, csv_path.name


# ----------------------------
# Parsing helpers
# ----------------------------
def parse_size_to_sqft(value):
    """
    Handles:
    - '1085 sqft'
    - '799-1258 sqft'
    - 'Varies'
    - 'Not specified'
    Returns (size_min_sqft, size_max_sqft, size_mean_sqft)
    """
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    s = str(value).strip().lower().replace(",", "")
    if s in {"varies", "various", "not specified", "na", "n/a", ""}:
        return np.nan, np.nan, np.nan

    nums = re.findall(r"(\d+(?:\.\d+)?)", s)
    if not nums:
        return np.nan, np.nan, np.nan

    nums = [float(x) for x in nums]

    if len(nums) == 1:
        return nums[0], nums[0], nums[0]

    size_min = min(nums)
    size_max = max(nums)
    size_mean = float(np.mean(nums))
    return size_min, size_max, size_mean


def parse_listing_date(value, reference_date=None):
    """
    Handles:
    - '2025-02-19'
    - '2025-02-19T04:38:38.179Z'
    - '1mo ago', '2w ago', '4d ago', 'Yesterday'
    relative dates are anchored to reference_date
    """
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip()
    s_lower = s.lower()

    # direct parse first
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.notna(dt):
        try:
            return dt.tz_convert(None)
        except Exception:
            try:
                return dt.tz_localize(None)
            except Exception:
                return pd.Timestamp(dt)

    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()

    if s_lower == "yesterday":
        return reference_date - pd.Timedelta(days=1)

    match = re.match(r"^\s*(\d+)\s*([dwm])\s*ago\s*$", s_lower)
    if match:
        qty = int(match.group(1))
        unit = match.group(2)
        if unit == "d":
            return reference_date - pd.Timedelta(days=qty)
        if unit == "w":
            return reference_date - pd.Timedelta(weeks=qty)
        if unit == "m":
            return reference_date - pd.Timedelta(days=30 * qty)

    return pd.NaT


def clean_property_group(type_value):
    if pd.isna(type_value):
        return "Unknown"

    s = str(type_value).strip().lower()

    if "plot" in s or "land" in s:
        return "Land/Plot"
    if "villa" in s:
        return "Villa"
    if "house" in s:
        return "House"
    if "apartment" in s or "flat" in s:
        return "Apartment/Flat"
    if "floor" in s:
        return "Independent Floor"
    return "Other"


# ----------------------------
# Data preparation
# ----------------------------
def basic_clean(df):
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df


def engineer_features(df):
    df = df.copy()

    # keep original values
    if "url" in df.columns:
        df["url"] = df["url"].astype(str).str.strip()

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip()

    if "neighborhood" in df.columns:
        df["neighborhood"] = df["neighborhood"].astype(str).str.strip()

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip()
        df["property_group"] = df["type"].apply(clean_property_group)
    else:
        df["property_group"] = "Unknown"

    # size parsing
    if "size" in df.columns:
        parsed = df["size"].apply(parse_size_to_sqft)
        parsed_df = pd.DataFrame(
            parsed.tolist(),
            columns=["size_min_sqft", "size_max_sqft", "size_mean_sqft"],
            index=df.index
        )
        df = pd.concat([df, parsed_df], axis=1)
    else:
        df["size_min_sqft"] = np.nan
        df["size_max_sqft"] = np.nan
        df["size_mean_sqft"] = np.nan

    # date parsing
    reference_date = None
    if "date" in df.columns:
        abs_dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if abs_dates.notna().any():
            reference_date = abs_dates.max().tz_convert(None).normalize()

        df["listing_date"] = df["date"].apply(lambda x: parse_listing_date(x, reference_date=reference_date))
        df["listing_age_days"] = (reference_date - df["listing_date"]).dt.days if reference_date is not None else np.nan
    else:
        df["listing_date"] = pd.NaT
        df["listing_age_days"] = np.nan

    # numeric enforcement
    for col in ["beds", "baths", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # price per sqft
    if "price" in df.columns:
        valid_area = df["size_mean_sqft"].replace(0, np.nan)
        df["price_per_sqft"] = df["price"] / valid_area
    else:
        df["price_per_sqft"] = np.nan

    # neighborhood bucketing to reduce cardinality
    if "neighborhood" in df.columns:
        top_n = 15
        top_neighborhoods = df["neighborhood"].value_counts().head(top_n).index
        df["neighborhood_bucket"] = np.where(
            df["neighborhood"].isin(top_neighborhoods),
            df["neighborhood"],
            "Other"
        )
    else:
        df["neighborhood_bucket"] = "Other"

    return df


# ----------------------------
# Quality reports
# ----------------------------
def build_data_quality_report(df):
    report = []

    literal_missing = df.isna().sum()

    for col in df.columns:
        report.append({
            "issue_type": "literal_missing",
            "column": col,
            "count": int(literal_missing[col])
        })

    if "url" in df.columns:
        report.append({
            "issue_type": "duplicate_url",
            "column": "url",
            "count": int(df["url"].duplicated().sum())
        })

    if "price" in df.columns:
        report.append({
            "issue_type": "non_positive_price",
            "column": "price",
            "count": int((df["price"].fillna(0) <= 0).sum())
        })

    if "size" in df.columns:
        report.append({
            "issue_type": "unparsed_size",
            "column": "size",
            "count": int(df["size_mean_sqft"].isna().sum())
        })
        report.append({
            "issue_type": "non_positive_size",
            "column": "size_mean_sqft",
            "count": int((df["size_mean_sqft"].fillna(0) <= 0).sum())
        })

    if "listing_date" in df.columns:
        report.append({
            "issue_type": "unparsed_date",
            "column": "listing_date",
            "count": int(df["listing_date"].isna().sum())
        })

    report_df = pd.DataFrame(report)
    report_df.to_csv("outputs/tables/data_quality_report.csv", index=False)
    return report_df


def flag_suspicious_records(df):
    flagged = df.copy()
    flagged["flag_non_positive_price"] = flagged["price"].fillna(0) <= 0
    flagged["flag_unparsed_size"] = flagged["size_mean_sqft"].isna()
    flagged["flag_non_positive_size"] = flagged["size_mean_sqft"].fillna(0) <= 0
    flagged["flag_duplicate_url"] = flagged["url"].duplicated(keep=False) if "url" in flagged.columns else False

    # price per sqft outliers
    valid_pps = flagged["price_per_sqft"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_pps) >= 10:
        q1 = valid_pps.quantile(0.25)
        q3 = valid_pps.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        flagged["flag_pps_outlier"] = (flagged["price_per_sqft"] < lower) | (flagged["price_per_sqft"] > upper)
    else:
        flagged["flag_pps_outlier"] = False

    flag_cols = [c for c in flagged.columns if c.startswith("flag_")]
    flagged["any_flag"] = flagged[flag_cols].any(axis=1)

    suspicious_df = flagged[flagged["any_flag"]].copy()
    suspicious_df.to_csv("outputs/tables/suspicious_records.csv", index=False)
    return suspicious_df


# ----------------------------
# EDA tables
# ----------------------------
def save_summary_tables(df):
    if "city" in df.columns:
        city_counts = df["city"].value_counts().reset_index()
        city_counts.columns = ["city", "listing_count"]
        city_counts.to_csv("outputs/tables/listings_by_city.csv", index=False)

    if {"city", "price"}.issubset(df.columns):
        city_price = (
            df.groupby("city", dropna=False)["price"]
            .agg(["count", "median", "mean"])
            .reset_index()
            .sort_values("median", ascending=False)
        )
        city_price.to_csv("outputs/tables/city_price_summary.csv", index=False)

    if {"property_group", "price"}.issubset(df.columns):
        type_price = (
            df.groupby("property_group", dropna=False)["price"]
            .agg(["count", "median", "mean"])
            .reset_index()
            .sort_values("median", ascending=False)
        )
        type_price.to_csv("outputs/tables/property_group_price_summary.csv", index=False)

    if "price_per_sqft" in df.columns and "city" in df.columns:
        pps_city = (
            df[df["price_per_sqft"].notna()]
            .groupby("city")["price_per_sqft"]
            .agg(["count", "median", "mean"])
            .reset_index()
            .sort_values("median", ascending=False)
        )
        pps_city.to_csv("outputs/tables/price_per_sqft_by_city.csv", index=False)


# ----------------------------
# Plots
# ----------------------------
def save_plot(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_data_quality(report_df):
    temp = report_df[report_df["count"] > 0].copy()
    if temp.empty:
        return

    temp = temp.sort_values("count", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(temp["issue_type"], temp["count"])
    plt.title("Data Quality Issues")
    plt.xlabel("Count")
    plt.ylabel("Issue Type")
    save_plot("outputs/figures/data_quality_issues.png")


def plot_price_distribution(df):
    if "price" not in df.columns:
        return

    clean_price = df["price"].dropna()
    if clean_price.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(clean_price, bins=35)
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    save_plot("outputs/figures/price_distribution.png")

    positive_price = clean_price[clean_price > 0]
    if len(positive_price) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(np.log1p(positive_price), bins=35)
        plt.title("Log Price Distribution")
        plt.xlabel("log(1 + Price)")
        plt.ylabel("Count")
        save_plot("outputs/figures/log_price_distribution.png")


def plot_city_listing_counts(df):
    if "city" not in df.columns:
        return

    temp = df["city"].value_counts().head(12)
    temp = temp.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(temp.index, temp.values)
    plt.title("Top Cities by Listing Count")
    plt.xlabel("Number of Listings")
    plt.ylabel("City")
    save_plot("outputs/figures/top_cities_by_listing_count.png")


def plot_median_price_by_city(df):
    if not {"city", "price"}.issubset(df.columns):
        return

    city_order = df["city"].value_counts().head(10).index
    temp = df[df["city"].isin(city_order)].copy()

    summary = (
        temp.groupby("city")["price"]
        .median()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(summary.index, summary.values)
    plt.title("Median Price by City (Top Cities)")
    plt.xlabel("Median Price")
    plt.ylabel("City")
    save_plot("outputs/figures/median_price_by_city.png")


def plot_price_by_property_group(df):
    if not {"property_group", "price"}.issubset(df.columns):
        return

    temp = df[df["price"].notna()].copy()
    if temp.empty:
        return

    groups = [grp["price"].values for _, grp in temp.groupby("property_group")]
    labels = [name for name, _ in temp.groupby("property_group")]

    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xticks(rotation=20)
    plt.title("Price by Property Group")
    plt.xlabel("Property Group")
    plt.ylabel("Price")
    save_plot("outputs/figures/price_by_property_group.png")


def plot_area_vs_price(df):
    needed = {"size_mean_sqft", "price"}
    if not needed.issubset(df.columns):
        return

    temp = df[df["size_mean_sqft"].notna() & (df["size_mean_sqft"] > 0) & df["price"].notna()].copy()
    if temp.empty:
        return

    size_cap = temp["size_mean_sqft"].quantile(0.99)
    temp = temp[temp["size_mean_sqft"] <= size_cap]

    plt.figure(figsize=(8, 5))
    plt.scatter(temp["size_mean_sqft"], temp["price"], alpha=0.7)
    plt.title("Area vs Price")
    plt.xlabel("Mean Size (sqft)")
    plt.ylabel("Price")
    save_plot("outputs/figures/area_vs_price.png")


def plot_price_per_sqft_by_city(df):
    if not {"city", "price_per_sqft"}.issubset(df.columns):
        return

    temp = df[df["price_per_sqft"].notna()].copy()
    if temp.empty:
        return

    top_cities = temp["city"].value_counts().head(10).index
    temp = temp[temp["city"].isin(top_cities)]

    summary = (
        temp.groupby("city")["price_per_sqft"]
        .median()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(summary.index, summary.values)
    plt.title("Median Price per Sqft by City")
    plt.xlabel("Median Price per Sqft")
    plt.ylabel("City")
    save_plot("outputs/figures/median_price_per_sqft_by_city.png")


def plot_listing_age_distribution(df):
    if "listing_age_days" not in df.columns:
        return

    temp = df["listing_age_days"].dropna()
    if temp.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(temp, bins=20)
    plt.title("Listing Age Distribution")
    plt.xlabel("Days Since Listing")
    plt.ylabel("Count")
    save_plot("outputs/figures/listing_age_distribution.png")


def plot_correlation_heatmap(df):
    numeric_cols = [
        c for c in ["beds", "baths", "price", "size_min_sqft", "size_max_sqft",
                    "size_mean_sqft", "price_per_sqft", "listing_age_days"]
        if c in df.columns
    ]
    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    if numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    save_plot("outputs/figures/correlation_heatmap.png")

# ----------------------------
# Modeling
# ----------------------------
def build_model_dataset(df):
    required_cols = ["price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing for modeling: {col}")

    model_df = df.copy()

    # only learn on rows with valid positive target
    model_df = model_df[model_df["price"].notna()].copy()
    model_df = model_df[model_df["price"] > 0].copy()

    # features chosen for this dataset
    feature_cols = [
        "beds",
        "baths",
        "city",
        "property_group",
        "type",
        "neighborhood_bucket",
        "size_min_sqft",
        "size_max_sqft",
        "size_mean_sqft",
        "listing_age_days",
    ]
    feature_cols = [c for c in feature_cols if c in model_df.columns]

    X = model_df[feature_cols].copy()
    y = model_df["price"].copy()

    return X, y


def train_model(X, y):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        min_samples_split=4,
        random_state=42,
        n_jobs=1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features_raw": int(X.shape[1])
    }

    with open("outputs/metrics/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipeline, "outputs/metrics/random_forest_pipeline.joblib")

    return pipeline, X_test, y_test, metrics


def save_feature_importance(pipeline, X_test, y_test):
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        n_jobs=1
    )

    importances = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    importances.to_csv("outputs/tables/feature_importance.csv", index=False)

    top_imp = importances.head(12)

    plt.figure(figsize=(10, 6))
    ordered = top_imp.sort_values("importance_mean", ascending=True)
    plt.barh(ordered["feature"], ordered["importance_mean"])
    plt.title("Top Feature Importances")
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    save_plot("outputs/figures/feature_importance.png")

    return importances


# ----------------------------
# Project summary
# ----------------------------
def save_project_summary(df, quality_report, suspicious_df, metrics):
    summary = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "city_count": int(df["city"].nunique()) if "city" in df.columns else None,
        "property_group_count": int(df["property_group"].nunique()) if "property_group" in df.columns else None,
        "duplicate_url_count": int(df["url"].duplicated().sum()) if "url" in df.columns else None,
        "unparsed_size_count": int(df["size_mean_sqft"].isna().sum()) if "size_mean_sqft" in df.columns else None,
        "non_positive_price_count": int((df["price"].fillna(0) <= 0).sum()) if "price" in df.columns else None,
        "suspicious_record_count": int(len(suspicious_df)),
        "model_metrics": metrics
    }

    with open("outputs/metrics/project_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ----------------------------
# Main
# ----------------------------
def main(input_path=None):
    ensure_dirs()

    df, filename = load_csv(input_path=input_path)
    print(f"[INFO] Raw shape: {df.shape}")

    df = basic_clean(df)
    df = engineer_features(df)

    # save processed dataset
    df.to_csv("data/processed/cleaned_dataset.csv", index=False)

    # reports
    quality_report = build_data_quality_report(df)
    suspicious_df = flag_suspicious_records(df)
    save_summary_tables(df)

    # plots
    plot_data_quality(quality_report)
    plot_price_distribution(df)
    plot_city_listing_counts(df)
    plot_median_price_by_city(df)
    plot_price_by_property_group(df)
    plot_area_vs_price(df)
    plot_price_per_sqft_by_city(df)
    plot_listing_age_distribution(df)
    plot_correlation_heatmap(df)

    # modeling
    X, y = build_model_dataset(df)
    pipeline, X_test, y_test, metrics = train_model(X, y)
    save_feature_importance(pipeline, X_test, y_test)

    # summary
    save_project_summary(df, quality_report, suspicious_df, metrics)

    print("\n[INFO] Analysis complete.")
    print("[INFO] Key metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    print("\n[INFO] Files created:")
    print("  - data/processed/cleaned_dataset.csv")
    print("  - outputs/tables/data_quality_report.csv")
    print("  - outputs/tables/suspicious_records.csv")
    print("  - outputs/tables/listings_by_city.csv")
    print("  - outputs/tables/city_price_summary.csv")
    print("  - outputs/tables/property_group_price_summary.csv")
    print("  - outputs/tables/price_per_sqft_by_city.csv")
    print("  - outputs/tables/feature_importance.csv")
    print("  - outputs/figures/*.png")
    print("  - outputs/metrics/model_metrics.json")
    print("  - outputs/metrics/project_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV file")
    args = parser.parse_args()
    main(input_path=args.input)
