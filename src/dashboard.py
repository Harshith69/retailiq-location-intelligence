"""
RetailIQ — Interactive Dashboard
=================================
Streamlit dashboard for exploring site scores, demand heatmaps,
network gaps, and generating per-location site reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import h3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Must be first Streamlit call
st.set_page_config(
    page_title="RetailIQ — Location Intelligence",
    page_icon="📍",
    layout="wide",
)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@st.cache_data
def load_data():
    """Load all processed data."""
    fm = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")
    scores = pd.read_parquet(DATA_DIR / "site_scores.parquet")

    # Merge
    df = fm.join(scores, rsuffix="_score")
    return df


def hex_to_geojson(hex_id):
    """Convert H3 hex to GeoJSON polygon."""
    boundary = h3.cell_to_boundary(hex_id)
    coords = [[lng, lat] for lat, lng in boundary]
    coords.append(coords[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"hex_id": hex_id},
    }


def score_color(score):
    """Map site score to color."""
    if score >= 85:
        return "#1a9850"   # A — green
    elif score >= 70:
        return "#91cf60"   # B — light green
    elif score >= 50:
        return "#fee08b"   # C — yellow
    elif score >= 30:
        return "#fc8d59"   # D — orange
    else:
        return "#d73027"   # E — red


def gap_color(gap):
    """Map gap score to color."""
    if gap >= 70:
        return "#08519c"   # Very High
    elif gap >= 50:
        return "#3182bd"   # High
    elif gap >= 30:
        return "#6baed6"   # Moderate
    else:
        return "#bdd7e7"   # Low


def render_site_score_map(df, center=(12.9716, 77.5946)):
    """Render H3 hexagonal heatmap colored by site score."""
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    for idx, row in df.iterrows():
        geojson = hex_to_geojson(idx)
        color = score_color(row["site_score"])

        folium.GeoJson(
            geojson,
            style_function=lambda x, c=color: {
                "fillColor": c,
                "color": "#333",
                "weight": 0.5,
                "fillOpacity": 0.65,
            },
            tooltip=folium.Tooltip(
                f"<b>Score: {row['site_score']:.1f}</b> (Grade {row['site_grade']})<br>"
                f"Demand: {row['demand_score']:.0f} | "
                f"Catchment: {row['catchment_score']:.0f}<br>"
                f"Access: {row['accessibility_score']:.0f} | "
                f"Competition: {row['competition_score']:.0f}<br>"
                f"POIs: {row.get('poi_total_count', 0):.0f} | "
                f"Pop proxy: {row.get('pop_density_proxy', 0):.0f}"
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:5px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:12px;">
        <b>Site Score</b><br>
        <span style="color:#1a9850">■</span> A (85–100)<br>
        <span style="color:#91cf60">■</span> B (70–85)<br>
        <span style="color:#fee08b">■</span> C (50–70)<br>
        <span style="color:#fc8d59">■</span> D (30–50)<br>
        <span style="color:#d73027">■</span> E (0–30)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def render_gap_map(df, center=(12.9716, 77.5946)):
    """Render network gap analysis map."""
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    for idx, row in df.iterrows():
        geojson = hex_to_geojson(idx)
        color = gap_color(row["gap_score"])

        folium.GeoJson(
            geojson,
            style_function=lambda x, c=color: {
                "fillColor": c,
                "color": "#333",
                "weight": 0.5,
                "fillOpacity": 0.6,
            },
            tooltip=folium.Tooltip(
                f"<b>Gap Score: {row['gap_score']:.1f}</b><br>"
                f"Demand: {row['demand_score']:.0f} | "
                f"Supply: {row.get('poi_total_count', 0):.0f} POIs<br>"
                f"Population proxy: {row.get('pop_density_proxy', 0):.0f}"
            ),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:5px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:12px;">
        <b>Network Gap</b><br>
        <span style="color:#08519c">■</span> Very High (70+)<br>
        <span style="color:#3182bd">■</span> High (50–70)<br>
        <span style="color:#6baed6">■</span> Moderate (30–50)<br>
        <span style="color:#bdd7e7">■</span> Low (0–30)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def render_demand_heatmap(df, center=(12.9716, 77.5946)):
    """Render demand estimation heatmap."""
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter")

    heat_data = [
        [row.center_lat, row.center_lon, row.demand_score / 100]
        for _, row in df.iterrows()
        if row.demand_score > 10
    ]

    HeatMap(
        heat_data,
        radius=18,
        blur=15,
        max_zoom=15,
        gradient={0.2: "blue", 0.4: "cyan", 0.6: "lime", 0.8: "yellow", 1.0: "red"},
    ).add_to(m)

    return m


def render_site_report(row, hex_id):
    """Render a detailed site report for a selected hexagon."""
    st.markdown(f"### 📍 Site Report: `{hex_id}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Site Score", f"{row['site_score']:.1f}", f"Grade {row['site_grade']}")
    col2.metric("Demand Score", f"{row['demand_score']:.0f}/100")
    col3.metric("Catchment Score", f"{row['catchment_score']:.0f}/100")
    col4.metric("Gap Score", f"{row['gap_score']:.0f}/100")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Score Breakdown")
        scores_data = {
            "Component": ["Demand", "Catchment", "Accessibility", "Competition (inv.)", "POI Diversity"],
            "Score": [
                row["demand_score"],
                row["catchment_score"],
                row["accessibility_score"],
                row["competition_score"],
                row["poi_diversity_score"],
            ],
            "Weight": ["30%", "25%", "15%", "15%", "15%"],
        }
        st.dataframe(pd.DataFrame(scores_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Location Features")
        features_data = {
            "Feature": [
                "Population proxy", "Total POIs", "Restaurants",
                "Retail stores", "Banks/ATMs", "Road density",
                "Catchment (3km)", "Competition index",
            ],
            "Value": [
                f"{row.get('pop_density_proxy', 0):,.0f}",
                f"{row.get('poi_total_count', 0):.0f}",
                f"{row.get('poi_restaurant_count', 0):.0f}",
                f"{row.get('poi_retail_count', 0):.0f}",
                f"{row.get('poi_bank_atm_count', 0):.0f}",
                f"{row.get('road_density', 0):,.0f}",
                f"{row.get('catchment_pop_3.0km', 0):,.0f}",
                f"{row.get('competition_index', 0):.3f}",
            ],
        }
        st.dataframe(pd.DataFrame(features_data), hide_index=True, use_container_width=True)

    # Score radar chart
    st.markdown("#### Score Radar")
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    categories = ["Demand", "Catchment", "Access.", "Compet.", "Diversity"]
    values = [
        row["demand_score"] / 100,
        row["catchment_score"] / 100,
        row["accessibility_score"] / 100,
        row["competition_score"] / 100,
        row["poi_diversity_score"] / 100,
    ]
    values += values[:1]  # close the polygon
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.fill(angles, values, alpha=0.25, color="#2563eb")
    ax.plot(angles, values, color="#2563eb", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8)
    ax.set_title(f"Score Profile — {hex_id[:12]}...", pad=20)
    st.pyplot(fig)
    plt.close()


def main():
    st.title("📍 RetailIQ — Location Intelligence for Retail Site Selection")
    st.markdown(
        "End-to-end spatial analytics platform for optimal retail network planning. "
        "Powered by H3 hexagonal grids, OpenStreetMap POI data, and gradient-boosted "
        "demand estimation."
    )

    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Minimum Site Score", 0, 100, 0)
    selected_grades = st.sidebar.multiselect(
        "Site Grades", ["A", "B", "C", "D", "E"],
        default=["A", "B", "C", "D", "E"],
    )

    # Filter data
    mask = (df["site_score"] >= min_score) & (df["site_grade"].isin(selected_grades))
    filtered = df[mask]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(filtered)}** / {len(df)} hexagons shown")
    st.sidebar.markdown(f"**Grade A sites:** {(df['site_grade'] == 'A').sum()}")
    st.sidebar.markdown(f"**Network gaps (High+):** {(df['gap_score'] >= 50).sum()}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Site Scores", "🔥 Demand Heatmap", "🕳️ Network Gaps",
        "📊 Analytics", "📋 Site Report",
    ])

    with tab1:
        st.subheader("Site Score Map")
        st.markdown("Hexagons colored by composite site score (A–E grading).")
        m = render_site_score_map(filtered)
        st_folium(m, width=None, height=550, returned_objects=[])

    with tab2:
        st.subheader("Demand Estimation Heatmap")
        st.markdown("Predicted demand intensity from the gradient-boosted regression model.")
        m = render_demand_heatmap(filtered)
        st_folium(m, width=None, height=550, returned_objects=[])

    with tab3:
        st.subheader("Network Gap Analysis")
        st.markdown(
            "Identifies underserved areas — high demand potential with low existing supply. "
            "Darker blue = bigger gap = better expansion opportunity."
        )
        m = render_gap_map(filtered)
        st_folium(m, width=None, height=550, returned_objects=[])

    with tab4:
        st.subheader("Analytics Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Hexagons", f"{len(df):,}")
        col2.metric("Avg Site Score", f"{df['site_score'].mean():.1f}")
        col3.metric("Grade A Sites", f"{(df['site_grade'] == 'A').sum()}")
        col4.metric("High Gap Areas", f"{(df['gap_score'] >= 50).sum()}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Site Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df["site_score"], bins=40, kde=True, ax=ax, color="#2563eb")
            ax.axvline(df["site_score"].mean(), color="red", linestyle="--",
                      label=f"Mean: {df['site_score'].mean():.1f}")
            ax.set_xlabel("Site Score")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Gap Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df["gap_score"], bins=40, kde=True, ax=ax, color="#08519c")
            ax.set_xlabel("Gap Score")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

        st.markdown("#### Feature Correlation Matrix")
        corr_cols = [
            "site_score", "demand_score", "catchment_score",
            "accessibility_score", "competition_score",
            "poi_diversity_score", "gap_score",
        ]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1,
        )
        ax.set_title("Score Component Correlations")
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Top 20 Sites")
        top_sites = df.nlargest(20, "site_score")[
            ["center_lat", "center_lon", "site_score", "site_grade",
             "demand_score", "catchment_score", "accessibility_score",
             "competition_score", "gap_score", "poi_total_count", "pop_density_proxy"]
        ].round(1)
        st.dataframe(top_sites, use_container_width=True)

    with tab5:
        st.subheader("Individual Site Report")
        st.markdown("Select a hexagon to view its detailed analysis.")

        # Sort by score for the dropdown
        sorted_hexes = df.nlargest(50, "site_score")
        options = [
            f"{idx} — Score: {row.site_score:.1f} (Grade {row.site_grade})"
            for idx, row in sorted_hexes.iterrows()
        ]

        if options:
            selected = st.selectbox("Select hexagon (top 50 by score):", options)
            hex_id = selected.split(" — ")[0]
            render_site_report(df.loc[hex_id], hex_id)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:12px;'>"
        "RetailIQ — Location Intelligence Platform | "
        "Built with H3, scikit-learn, Folium, Streamlit | "
        "Data: Synthetic (Bangalore spatial patterns) — "
        "production pipeline supports OSM Overpass API + Kontur Population</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
