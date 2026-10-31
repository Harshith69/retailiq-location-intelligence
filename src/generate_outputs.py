"""
RetailIQ — Static Output Generator
====================================
Generates standalone HTML maps and PNG charts for the portfolio.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import h3
import folium
from folium.plugins import HeatMap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
import joblib

from config import CITY_CENTER

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def hex_to_geojson(hex_id):
    boundary = h3.cell_to_boundary(hex_id)
    coords = [[lng, lat] for lat, lng in boundary]
    coords.append(coords[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"hex_id": hex_id},
    }


def score_color(score):
    if score >= 85: return "#1a9850"
    elif score >= 70: return "#91cf60"
    elif score >= 50: return "#fee08b"
    elif score >= 30: return "#fc8d59"
    else: return "#d73027"


def gap_color(gap):
    if gap >= 70: return "#08519c"
    elif gap >= 50: return "#3182bd"
    elif gap >= 30: return "#6baed6"
    else: return "#bdd7e7"


def generate_site_score_map(df):
    print("  Generating site score map...")
    m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="OpenStreetMap")

    for idx, row in df.iterrows():
        geojson = hex_to_geojson(idx)
        color = score_color(row["site_score"])
        folium.GeoJson(
            geojson,
            style_function=lambda x, c=color: {
                "fillColor": c, "color": "#333",
                "weight": 0.5, "fillOpacity": 0.65,
            },
            tooltip=folium.Tooltip(
                f"<b>Score: {row['site_score']:.1f}</b> (Grade {row['site_grade']})<br>"
                f"Demand: {row['demand_score']:.0f} | Catchment: {row['catchment_score']:.0f}<br>"
                f"POIs: {row.get('poi_total_count', 0):.0f}"
            ),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:13px;">
        <b>Site Score</b><br>
        <span style="color:#1a9850;font-size:16px">■</span> A (85–100)<br>
        <span style="color:#91cf60;font-size:16px">■</span> B (70–85)<br>
        <span style="color:#fee08b;font-size:16px">■</span> C (50–70)<br>
        <span style="color:#fc8d59;font-size:16px">■</span> D (30–50)<br>
        <span style="color:#d73027;font-size:16px">■</span> E (0–30)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(OUTPUT_DIR / "site_score_map.html"))
    print("    → Saved: site_score_map.html")


def generate_gap_map(df):
    print("  Generating network gap map...")
    m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="OpenStreetMap")

    for idx, row in df.iterrows():
        geojson = hex_to_geojson(idx)
        color = gap_color(row["gap_score"])
        folium.GeoJson(
            geojson,
            style_function=lambda x, c=color: {
                "fillColor": c, "color": "#333",
                "weight": 0.5, "fillOpacity": 0.6,
            },
            tooltip=folium.Tooltip(
                f"<b>Gap Score: {row['gap_score']:.1f}</b><br>"
                f"Demand: {row['demand_score']:.0f} | POIs: {row.get('poi_total_count', 0):.0f}"
            ),
        ).add_to(m)

    m.save(str(OUTPUT_DIR / "network_gap_map.html"))
    print("    → Saved: network_gap_map.html")


def generate_demand_heatmap(df):
    print("  Generating demand heatmap...")
    m = folium.Map(location=CITY_CENTER, zoom_start=12, tiles="OpenStreetMap")

    heat_data = [
        [row.center_lat, row.center_lon, row.demand_score / 100]
        for _, row in df.iterrows() if row.demand_score > 10
    ]
    HeatMap(heat_data, radius=18, blur=15, max_zoom=15,
            gradient={0.2: "blue", 0.4: "cyan", 0.6: "lime", 0.8: "yellow", 1.0: "red"}
            ).add_to(m)

    m.save(str(OUTPUT_DIR / "demand_heatmap.html"))
    print("    → Saved: demand_heatmap.html")


def generate_analytics_charts(df):
    print("  Generating analytics charts...")
    sns.set_style("whitegrid")

    # 1. Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["site_score"], bins=40, kde=True, ax=axes[0], color="#2563eb")
    axes[0].axvline(df["site_score"].mean(), color="red", linestyle="--",
                   label=f'Mean: {df["site_score"].mean():.1f}')
    axes[0].set_title("Site Score Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Site Score")
    axes[0].legend()

    sns.histplot(df["gap_score"], bins=40, kde=True, ax=axes[1], color="#08519c")
    axes[1].set_title("Network Gap Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Gap Score")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved: score_distributions.png")

    # 2. Correlation matrix
    corr_cols = [
        "site_score", "demand_score", "catchment_score",
        "accessibility_score", "competition_score",
        "poi_diversity_score", "gap_score",
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Score Component Correlations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved: correlation_matrix.png")

    # 3. Feature importance
    try:
        model = joblib.load(OUTPUT_DIR / "demand_model.pkl")
        feature_cols = [
            "pop_density_proxy", "poi_restaurant_count", "poi_retail_count",
            "poi_bank_atm_count", "poi_education_count", "poi_healthcare_count",
            "poi_transport_count", "poi_fuel_count", "poi_entertainment_count",
            "poi_total_count", "catchment_pop_0.5km", "catchment_pop_1.0km",
            "catchment_pop_2.0km", "catchment_pop_3.0km", "catchment_pop_5.0km",
            "road_density", "weighted_road_access", "competition_index",
            "poi_diversity_score", "dist_to_center_km", "building_count", "avg_levels",
        ]
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        importance = importance.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        importance.plot(kind="barh", ax=ax, color="#2563eb")
        ax.set_title("Top 15 Features — Demand Estimation Model",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    → Saved: feature_importance.png")
    except Exception as e:
        print(f"    ⚠ Could not generate feature importance: {e}")

    # 4. Grade breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    grade_counts = df["site_grade"].value_counts().sort_index()
    colors = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
    grade_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Site Grade Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Number of Hexagons")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(grade_counts):
        ax.text(i, v + 5, str(v), ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "grade_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    → Saved: grade_distribution.png")


def main():
    print(f"\n{'='*60}")
    print("RetailIQ — Generating Static Outputs")
    print(f"{'='*60}\n")

    # Load data
    fm = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")
    scores = pd.read_parquet(DATA_DIR / "site_scores.parquet")
    df = fm.join(scores, rsuffix="_score")

    # Generate outputs
    generate_site_score_map(df)
    generate_gap_map(df)
    generate_demand_heatmap(df)
    generate_analytics_charts(df)

    print(f"\n{'='*60}")
    print("All outputs generated!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
