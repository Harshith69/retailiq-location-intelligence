"""
RetailIQ — Demand Estimation & Site Scoring
============================================
Mirrors GeoIQ's approach:
  1. Demand estimation: Gradient Boosted regression trained on
     simulated store performance data (in production, trained on
     actual revenue per store).
  2. Network gap analysis: Identifies underserved hexagons by
     comparing demand potential vs. existing supply.
  3. Composite site scoring: Weighted 0–100 score combining
     demand, catchment, accessibility, competition, and diversity.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

from config import SCORE_WEIGHTS

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "outputs"


def generate_training_labels(feature_matrix: pd.DataFrame) -> pd.Series:
    """
    Generate synthetic revenue proxy labels for training.

    In production, this would be actual store-level revenue data
    from existing locations (the approach GeoIQ uses with Lenskart).
    Here we simulate it using a known function of features + noise,
    so the model can learn the spatial demand signal.
    """
    print("  Generating training labels (revenue proxy)...")
    fm = feature_matrix.copy()

    # Revenue proxy = weighted combination of features + noise
    # This mirrors GeoIQ's regression approach from the Lenskart case study
    revenue_proxy = (
        0.25 * fm["pop_density_proxy"] / fm["pop_density_proxy"].max() +
        0.20 * fm["catchment_pop_3.0km"] / fm["catchment_pop_3.0km"].max() +
        0.15 * fm["poi_total_count"] / fm["poi_total_count"].max() +
        0.10 * fm["road_density"] / fm["road_density"].max() +
        0.10 * fm["poi_diversity_score"] +
        0.10 * (1 - fm["competition_index"]) +  # less competition = more revenue
        0.10 * (1 - fm["dist_to_center_km"] / fm["dist_to_center_km"].max())
    )

    # Scale to realistic monthly revenue range (₹2L - ₹25L)
    revenue = 200000 + revenue_proxy * 2300000

    # Add realistic noise (±15%)
    noise = np.random.normal(1.0, 0.15, len(revenue))
    revenue = revenue * noise
    revenue = revenue.clip(lower=50000)

    print(f"    → Revenue range: ₹{revenue.min():,.0f} — ₹{revenue.max():,.0f}")
    print(f"    → Median: ₹{revenue.median():,.0f}")
    return revenue


def train_demand_model(feature_matrix: pd.DataFrame) -> dict:
    """
    Train a Gradient Boosted Regression model for demand estimation.

    Returns dict with:
      - model: trained GBR model
      - scaler: fitted MinMaxScaler
      - metrics: train/test performance
      - feature_importance: ranked feature importances
      - predictions: predicted demand for all hexagons
    """
    print(f"\n{'='*60}")
    print("RetailIQ Demand Estimation Model")
    print(f"{'='*60}\n")

    fm = feature_matrix.copy()
    revenue = generate_training_labels(fm)

    # Feature columns
    feature_cols = [
        "pop_density_proxy", "poi_restaurant_count", "poi_retail_count",
        "poi_bank_atm_count", "poi_education_count", "poi_healthcare_count",
        "poi_transport_count", "poi_fuel_count", "poi_entertainment_count",
        "poi_total_count", "catchment_pop_0.5km", "catchment_pop_1.0km",
        "catchment_pop_2.0km", "catchment_pop_3.0km", "catchment_pop_5.0km",
        "road_density", "weighted_road_access", "competition_index",
        "poi_diversity_score", "dist_to_center_km", "building_count",
        "avg_levels",
    ]

    X = fm[feature_cols].values
    y = revenue.values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Gradient Boosted Regression
    print("[1/3] Training Gradient Boosted Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n[2/3] Model Evaluation")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")

    metrics = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    }

    print(f"  Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}")
    print(f"  Train MAE: ₹{train_mae:,.0f}  |  Test MAE: ₹{test_mae:,.0f}")
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    print("\n[3/3] Feature Importance")
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("  Top 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:35s} {imp:.4f}")

    # Full predictions
    all_predictions = model.predict(X_scaled)
    fm["predicted_demand"] = all_predictions

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "demand_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": importance,
        "predictions": fm["predicted_demand"],
        "feature_cols": feature_cols,
    }


def network_gap_analysis(feature_matrix: pd.DataFrame,
                          demand_predictions: pd.Series) -> pd.DataFrame:
    """
    Network gap analysis: find underserved hexagons.

    Gap score = demand potential - existing supply intensity.
    High gap → promising location for new stores.
    """
    print(f"\n{'='*60}")
    print("Network Gap Analysis")
    print(f"{'='*60}\n")

    fm = feature_matrix.copy()
    fm["predicted_demand"] = demand_predictions

    # Supply intensity: normalized POI count (existing commercial presence)
    supply = fm["poi_total_count"] / fm["poi_total_count"].max()

    # Demand potential: normalized predicted demand
    demand = fm["predicted_demand"] / fm["predicted_demand"].max()

    # Gap = high demand, low supply
    fm["demand_norm"] = demand
    fm["supply_norm"] = supply
    fm["gap_score"] = demand - supply * 0.7  # weight supply lower

    # Normalize gap to 0-1
    gap_min = fm["gap_score"].min()
    gap_max = fm["gap_score"].max()
    fm["gap_score"] = (fm["gap_score"] - gap_min) / (gap_max - gap_min)

    # Classify gaps
    fm["gap_category"] = pd.cut(
        fm["gap_score"],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["Low", "Moderate", "High", "Very High"],
    )

    top_gaps = fm.nlargest(20, "gap_score")
    print(f"  Top 20 underserved hexagons:")
    print(f"  {'Hex ID':20s} {'Gap Score':>10s} {'Demand':>12s} {'Supply':>10s}")
    print(f"  {'-'*55}")
    for _, row in top_gaps.head(10).iterrows():
        print(f"  {row.name:20s} {row.gap_score:10.3f} "
              f"₹{row.predicted_demand:10,.0f} {row.supply_norm:10.3f}")

    gap_counts = fm["gap_category"].value_counts()
    print(f"\n  Gap distribution:")
    for cat in ["Very High", "High", "Moderate", "Low"]:
        if cat in gap_counts.index:
            print(f"    {cat:12s}: {gap_counts[cat]:5d} hexagons")

    return fm[["demand_norm", "supply_norm", "gap_score", "gap_category"]]


def compute_site_scores(feature_matrix: pd.DataFrame,
                         demand_predictions: pd.Series,
                         gap_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite site score (0–100) for each hexagon.

    Weights (from config):
      demand_score:      30%
      catchment_score:   25%
      accessibility:     15%
      competition:       15% (inverse — lower competition = better)
      poi_diversity:     15%
    """
    print(f"\n{'='*60}")
    print("Composite Site Scoring")
    print(f"{'='*60}\n")

    fm = feature_matrix.copy()
    fm["predicted_demand"] = demand_predictions
    fm = fm.join(gap_analysis[["gap_score"]])

    scores = pd.DataFrame(index=fm.index)

    # 1. Demand score (0-100)
    demand_max = fm["predicted_demand"].quantile(0.99)
    scores["demand_score"] = np.clip(fm["predicted_demand"] / demand_max * 100, 0, 100)

    # 2. Catchment score (0-100)
    catch_max = fm["catchment_pop_3.0km"].quantile(0.99)
    scores["catchment_score"] = np.clip(fm["catchment_pop_3.0km"] / catch_max * 100, 0, 100)

    # 3. Accessibility score (0-100)
    road_max = fm["road_density"].quantile(0.99)
    scores["accessibility_score"] = np.clip(fm["road_density"] / road_max * 100, 0, 100)

    # 4. Competition score (inverse: less competition = higher score)
    scores["competition_score"] = (1 - fm["competition_index"]) * 100

    # 5. POI diversity (0-100)
    scores["poi_diversity_score"] = fm["poi_diversity_score"] * 100

    # Composite weighted score
    composite = sum(
        scores[component] * weight
        for component, weight in SCORE_WEIGHTS.items()
    )
    scores["site_score"] = composite

    # Classification
    scores["site_grade"] = pd.cut(
        scores["site_score"],
        bins=[0, 30, 50, 70, 85, 100],
        labels=["E", "D", "C", "B", "A"],
    )

    scores["gap_score"] = gap_analysis["gap_score"] * 100

    # Summary
    print(f"  Score distribution:")
    grade_counts = scores["site_grade"].value_counts().sort_index()
    for grade in ["A", "B", "C", "D", "E"]:
        if grade in grade_counts.index:
            print(f"    Grade {grade}: {grade_counts[grade]:5d} hexagons")

    print(f"\n  Top 10 sites:")
    top = scores.nlargest(10, "site_score")
    print(f"  {'Hex ID':20s} {'Score':>7s} {'Grade':>6s} {'Demand':>8s} "
          f"{'Catch.':>8s} {'Access':>8s} {'Comp.':>8s}")
    print(f"  {'-'*75}")
    for idx, row in top.iterrows():
        print(f"  {idx:20s} {row.site_score:7.1f} {row.site_grade:>6s} "
              f"{row.demand_score:8.1f} {row.catchment_score:8.1f} "
              f"{row.accessibility_score:8.1f} {row.competition_score:8.1f}")

    # Save
    scores.to_parquet(DATA_DIR / "processed" / "site_scores.parquet")
    print(f"\n  Saved: site_scores.parquet")

    return scores


def run_modeling(feature_matrix: pd.DataFrame) -> dict:
    """Run the full modeling pipeline."""
    # 1. Train demand model
    model_result = train_demand_model(feature_matrix)

    # 2. Network gap analysis
    gap = network_gap_analysis(feature_matrix, model_result["predictions"])

    # 3. Composite site scoring
    scores = compute_site_scores(
        feature_matrix, model_result["predictions"], gap
    )

    return {
        "model_result": model_result,
        "gap_analysis": gap,
        "site_scores": scores,
    }


if __name__ == "__main__":
    fm = pd.read_parquet(DATA_DIR / "processed" / "feature_matrix.parquet")
    results = run_modeling(fm)
