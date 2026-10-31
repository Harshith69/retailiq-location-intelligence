# 📍 RetailIQ — Location Intelligence for Retail Site Selection

> **An end-to-end spatial analytics platform that scores every 460m patch of a city for retail expansion potential — powered by H3 hexagonal grids, OpenStreetMap data, and gradient-boosted demand estimation.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![H3](https://img.shields.io/badge/Uber_H3-Hexagonal_Grid-1d9bf0)](https://h3geo.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

Retail chains expanding into a city face a critical question: **where should the next store go?**

Getting this wrong is expensive — a poorly located store bleeds money for years. Getting it right requires synthesizing population density, commercial activity, road access, competitive landscape, and catchment potential into a single decision.

**RetailIQ solves this.** It ingests open geospatial data, indexes it to a unified hexagonal grid, engineers 26 spatial features per location, trains a demand estimation model, and outputs a composite 0–100 site score with A–E grading for every micro-location in the city.

---

## What It Does

<table>
<tr>
<td width="50%">

### 🗺️ Site Score Map
Every hexagon scored and graded. Green (A) = prime location, red (E) = avoid. Hover for full score breakdown.

</td>
<td width="50%">

### 🔥 Demand Heatmap
Predicted revenue potential from the gradient-boosted regression model. Hot zones are where money flows.

</td>
</tr>
<tr>
<td>

### 🕳️ Network Gap Analysis
Where demand is high but supply is low — the white spaces on the map where a new store would dominate.

</td>
<td>

### 📊 Per-Site Reports
Radar charts, feature breakdowns, and competitive benchmarks for any selected hexagon.

</td>
</tr>
</table>

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ OSM POIs │  │ Kontur Pop.  │  │ Census    │  │ Road Network │  │
│  │ 8 types  │  │ H3 density   │  │ 2011 demo │  │ OSM ways     │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └──────┬───────┘  │
└───────┼───────────────┼────────────────┼───────────────┼───────────┘
        │               │                │               │
        ▼               ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    H3 HEXAGONAL GRID (Res 8 — ~460m)               │
│              All data indexed to 1,237 unified hex cells            │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                            │
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐ ┌──────────────┐   │
│  │ POI density│ │ Catchment  │ │ Competition │ │ Accessibility│   │
│  │ (8 types)  │ │ (5 rings)  │ │ index       │ │ score        │   │
│  └────────────┘ └────────────┘ └─────────────┘ └──────────────┘   │
│  → 26 features per hexagon                                        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ANALYTICS MODELS                              │
│  ┌──────────────┐  ┌────────────────┐  ┌───────────────────────┐   │
│  │ GBR Demand   │  │ Network Gap    │  │ Composite Site Score  │   │
│  │ Estimation   │  │ Analysis       │  │ (0–100, Grade A–E)    │   │
│  │ CV R²=0.80   │  │ Demand-Supply  │  │ Weighted: 5 components│   │
│  └──────────────┘  └────────────────┘  └───────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Interactive  │  │ Site Reports │  │ Streamlit Dashboard      │  │
│  │ Folium Maps  │  │ Per-location │  │ 5 tabs, live filtering   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/harshith69/retailiq-location-intelligence.git
cd retailiq-location-intelligence

# Install
pip install -r requirements.txt

# Run full pipeline (data → features → model → scores)
python run_pipeline.py

# Launch interactive dashboard
streamlit run src/dashboard.py

# Use live OpenStreetMap data (requires internet)
python run_pipeline.py --live
```

---

## Project Structure

```
retailiq-location-intelligence/
│
├── README.md                       ← You are here
├── requirements.txt                ← Python dependencies
├── run_pipeline.py                 ← One-command pipeline runner
│
├── src/
│   ├── config.py                   ← All parameters: city bounds, H3 res,
│   │                                  POI categories, score weights
│   ├── data_ingestion.py           ← H3 grid generation + OSM/synthetic data
│   ├── feature_engineering.py      ← 26 spatial features per hexagon
│   ├── models.py                   ← GBR demand model + gap analysis + scoring
│   ├── dashboard.py                ← Streamlit dashboard (5 tabs)
│   └── generate_outputs.py         ← Static HTML maps + PNG charts
│
├── data/
│   ├── raw/                        ← Ingested POIs, buildings, roads (GeoJSON)
│   └── processed/                  ← H3 grid, feature matrix, scores (Parquet)
│
└── outputs/
    ├── site_score_map.html         ← Interactive site score map
    ├── network_gap_map.html        ← Network gap analysis map
    ├── demand_heatmap.html         ← Demand estimation heatmap
    ├── feature_importance.png      ← Top model features
    ├── grade_distribution.png      ← A–E grade breakdown
    ├── score_distributions.png     ← Score & gap histograms
    ├── correlation_matrix.png      ← Score component correlations
    ├── demand_model.pkl            ← Trained GBR model
    └── scaler.pkl                  ← Fitted MinMaxScaler
```

---

## Technical Deep Dive

### 1. Spatial Indexing — H3 Hexagonal Grid

The entire city is tiled into **1,237 hexagonal cells** at H3 resolution 8 (~460m edge length). Hexagons outperform square grids for spatial analytics because:
- Every neighbor is equidistant (no diagonal bias)
- Catchment rings are uniform in all directions
- Hierarchical — zoom in/out by changing resolution
- Industry standard (used by Uber, Kontur, GeoIQ)

```python
# Generate all H3 cells covering Bangalore
hex_ids = set()
for lat in np.arange(12.85, 13.10, 0.002):
    for lon in np.arange(77.45, 77.75, 0.002):
        hex_ids.add(h3.latlng_to_cell(lat, lon, 8))
# → 1,237 hexagons
```

### 2. Feature Engineering — 26 Dimensions per Hexagon

| # | Feature Group | Features | Method |
|---|---|---|---|
| 1 | Population proxy | `pop_density_proxy`, `building_count`, `avg_levels` | Building footprint count × floor levels |
| 2 | POI density | 8 category counts + `poi_total_count` | Spatial join: POI → containing hexagon |
| 3 | Catchment population | 5 ring sums (0.5–5 km) | H3 `grid_disk(k)` neighbor aggregation |
| 4 | Road accessibility | `road_density`, `weighted_road_access` | Road clipping to hex + type weighting |
| 5 | Competition | `competition_index` | POI count ÷ population (market saturation) |
| 6 | Diversity | `poi_diversity_score` | Shannon entropy across 8 POI categories |
| 7 | Centrality | `dist_to_center_km` | Haversine distance to CBD |

### 3. Demand Estimation — Gradient Boosted Regression

The model predicts revenue potential per hexagon, mirroring how GeoIQ trains regression models on actual store performance data (as documented in their Lenskart case study).

```
Model:              GradientBoostingRegressor
Estimators:         200 trees, max_depth=5, lr=0.1
Train R²:           0.9816
Test R²:            0.7607
5-Fold CV R²:       0.8005 ± 0.0173
Test MAE:           ₹123,850
```

**Top features by importance:**
1. `catchment_pop_2km` — 51.2%
2. `catchment_pop_5km` — 9.5%
3. `catchment_pop_1km` — 7.8%
4. `catchment_pop_0.5km` — 5.9%
5. `dist_to_center_km` — 4.1%

Catchment population dominates — the number of people reachable within 2 km is the single strongest predictor of retail demand.

### 4. Network Gap Analysis

Identifies underserved locations by comparing predicted demand against existing commercial supply:

```
Gap Score = Normalized Demand − 0.7 × Normalized Supply
```

| Gap Category | Hexagons | Interpretation |
|---|---|---|
| Very High | 72 | Prime expansion targets |
| High | 505 | Strong opportunity zones |
| Moderate | 634 | Established but room to grow |
| Low | 25 | Saturated — avoid |

### 5. Composite Site Scoring

Each hexagon gets a **0–100 score** from five weighted components:

| Component | Weight | What It Captures |
|---|---|---|
| Demand Score | 30% | Predicted revenue potential |
| Catchment Score | 25% | Population within 3 km |
| Accessibility Score | 15% | Road network density |
| Competition Score (inverse) | 15% | Lower competition = higher score |
| POI Diversity Score | 15% | Ecosystem health (Shannon entropy) |

**Grade distribution:**
| Grade | Score Range | Count | Meaning |
|---|---|---|---|
| A | 85–100 | 18 | Best locations in the city |
| B | 70–85 | 87 | Strong locations |
| C | 50–70 | 236 | Average — viable with strategy |
| D | 30–50 | 666 | Below average |
| E | 0–30 | 230 | Avoid |

---

## Data Sources

| Source | What | Resolution | Usage |
|---|---|---|---|
| **OpenStreetMap** (Overpass API) | POIs, buildings, roads | Point / line | Commercial activity, population proxy, accessibility |
| **Kontur Population** | Population density | H3 res-8 (~400m) | Catchment population (production) |
| **India Census 2011** | Demographics | Ward / district | Baseline demographics |

The pipeline ships with a **synthetic data generator** that mirrors real Bangalore spatial patterns (POI clustering around Koramangala, Indiranagar, Whitefield, etc.) for demonstration. The `--live` flag activates the Overpass API pipeline for real OSM data.

---

## Streamlit Dashboard

The interactive dashboard has 5 tabs:

1. **🗺️ Site Scores** — H3 hexagonal map colored by composite score (A–E)
2. **🔥 Demand Heatmap** — Predicted demand intensity visualization
3. **🕳️ Network Gaps** — Underserved areas for expansion targeting
4. **📊 Analytics** — Score distributions, correlations, top sites table
5. **📋 Site Report** — Per-hexagon deep dive with radar chart

```bash
streamlit run src/dashboard.py
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Spatial** | H3-py, GeoPandas, Shapely, Folium |
| **Data** | Pandas, NumPy, Parquet, GeoJSON |
| **ML** | scikit-learn (GBR), MinMaxScaler, cross_val_score |
| **Viz** | Matplotlib, Seaborn, Folium HeatMap |
| **App** | Streamlit, streamlit-folium |
| **Data sources** | OpenStreetMap (Overpass API), Kontur |

---

## Extending the Project

RetailIQ is designed to be modular. Here's how to extend it:

**Add a new city:**
```python
# In config.py — change the bounding box and center
CITY_NAME = "Mumbai"
CITY_BBOX = {"south": 18.88, "north": 19.28, "west": 72.77, "east": 72.98}
CITY_CENTER = (19.0760, 72.8777)
```

**Add a new data source:**
```python
# Any source that outputs (lat, lon, category) can be plugged in
# Just add a new function in data_ingestion.py that returns a GeoDataFrame
def fetch_from_google_places(api_key: str) -> gpd.GeoDataFrame:
    # ... fetch and return same schema as fetch_pois()
```

**Add a new feature:**
```python
# Add computation in feature_engineering.py, join to the feature matrix
# The model retrains automatically on the expanded feature set
```

**Swap in real revenue data:**
```python
# Replace generate_training_labels() in models.py with actual
# store-level revenue data — the rest of the pipeline stays identical
```

---

## Author

**Harshith N Chandan**
Data Scientist & ML Engineer | Geospatial Analytics

---

## License

MIT License — see [LICENSE](LICENSE) for details.
