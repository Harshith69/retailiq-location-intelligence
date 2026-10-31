"""
RetailIQ — Data Ingestion
=========================
Generates H3 hexagonal grid for the target city and loads spatial data.

In production, POI data comes from OpenStreetMap (Overpass API) and
population from Kontur. This module includes a synthetic data generator
that mimics real Bangalore spatial patterns for demonstration,
with the full Overpass pipeline ready to swap in.
"""

import json
import time
import requests
import pandas as pd
import geopandas as gpd
import h3
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from pathlib import Path

from config import (
    CITY_BBOX, CITY_CENTER, CITY_NAME,
    H3_RESOLUTION, POI_CATEGORIES, OVERPASS_URL,
)

DATA_DIR = Path(__file__).parent.parent / "data"
np.random.seed(42)


# ---------------------------------------------------------------------------
# H3 grid generation (always runs — no external dependency)
# ---------------------------------------------------------------------------

def generate_h3_grid() -> gpd.GeoDataFrame:
    """
    Generate H3 hexagonal grid covering the city bounding box.
    Returns a GeoDataFrame with hex_id and polygon geometry.
    """
    bbox = CITY_BBOX
    print(f"  Generating H3 grid (resolution {H3_RESOLUTION})...")

    lats = np.arange(bbox["south"], bbox["north"], 0.002)
    lons = np.arange(bbox["west"], bbox["east"], 0.002)

    hex_ids = set()
    for lat in lats:
        for lon in lons:
            hex_ids.add(h3.latlng_to_cell(lat, lon, H3_RESOLUTION))

    hexagons = []
    for hid in hex_ids:
        boundary = h3.cell_to_boundary(hid)
        poly = Polygon([(lng, lat) for lat, lng in boundary])
        center = h3.cell_to_latlng(hid)
        hexagons.append({
            "hex_id": hid,
            "center_lat": center[0],
            "center_lon": center[1],
            "geometry": poly,
        })

    gdf = gpd.GeoDataFrame(hexagons, crs="EPSG:4326")
    print(f"    → {len(gdf)} hexagons")
    return gdf


# ---------------------------------------------------------------------------
# Synthetic data generators — mirror real Bangalore spatial patterns
# ---------------------------------------------------------------------------

def _distance_to_center(lat, lon):
    """Haversine distance (km) from a point to city center."""
    R = 6371
    dlat = np.radians(lat - CITY_CENTER[0])
    dlon = np.radians(lon - CITY_CENTER[1])
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(CITY_CENTER[0])) *
         np.cos(np.radians(lat)) *
         np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Known commercial hotspots in Bangalore (lat, lon, radius_km, intensity)
_HOTSPOTS = [
    (12.9716, 77.5946, 2.0, 1.0),    # MG Road / Central
    (12.9352, 77.6245, 1.5, 0.9),    # Koramangala
    (12.9698, 77.7500, 1.5, 0.8),    # Whitefield
    (12.9784, 77.6408, 1.2, 0.85),   # Indiranagar
    (13.0358, 77.5970, 1.2, 0.7),    # Hebbal / Yelahanka
    (12.9063, 77.5857, 1.0, 0.75),   # Jayanagar / JP Nagar
    (12.9165, 77.6101, 1.0, 0.7),    # BTM Layout
    (13.0012, 77.5437, 1.2, 0.6),    # Rajajinagar
    (12.9568, 77.7009, 1.0, 0.65),   # Marathahalli
    (13.0632, 77.5857, 0.8, 0.5),    # Yelahanka New Town
    (12.9141, 77.6366, 1.2, 0.8),    # HSR Layout
]


def _density_at(lat, lon):
    """
    Compute a density weight at (lat, lon) based on distance from
    city center and proximity to known commercial hotspots.
    """
    # Base density: exponential decay from center
    d_center = _distance_to_center(lat, lon)
    base = np.exp(-0.15 * d_center)

    # Hotspot boost
    hotspot_boost = 0
    for hlat, hlon, hradius, hintensity in _HOTSPOTS:
        d = _distance_to_center(lat - hlat + CITY_CENTER[0],
                                 lon - hlon + CITY_CENTER[1])
        # Use actual distance to hotspot
        dlat = lat - hlat
        dlon = lon - hlon
        d_hotspot = np.sqrt(dlat**2 + dlon**2) * 111  # rough km
        if d_hotspot < hradius * 2:
            hotspot_boost = max(hotspot_boost,
                                hintensity * np.exp(-1.5 * d_hotspot / hradius))

    return np.clip(base + hotspot_boost, 0, 1)


def generate_synthetic_pois(n_total: int = 8000) -> gpd.GeoDataFrame:
    """
    Generate spatially realistic POI data mirroring Bangalore's
    commercial geography. POIs cluster around known commercial areas
    with category-appropriate density patterns.
    """
    print("  Generating synthetic POIs (Bangalore spatial pattern)...")
    bbox = CITY_BBOX

    # Category distribution (mirrors typical Indian metro)
    cat_weights = {
        "restaurant": 0.28,
        "retail": 0.22,
        "bank_atm": 0.12,
        "education": 0.08,
        "healthcare": 0.10,
        "entertainment": 0.04,
        "transport": 0.06,
        "fuel": 0.10,
    }

    # Sample names per category
    sample_names = {
        "restaurant": ["Darshini", "Meghana Foods", "Cafe Artisan", "MTR", "Empire",
                       "Udupi Garden", "Nandhini", "Taco Bell", "Pizza Corner",
                       "Barbeque Nation", "Chai Point", "Third Wave Coffee"],
        "retail": ["Reliance Smart", "More Supermarket", "BigBasket Store", "DMart",
                   "Spar Hypermarket", "Westside", "Lifestyle", "Central Mall",
                   "Shoppers Stop", "Allen Solly", "Peter England"],
        "bank_atm": ["SBI ATM", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak ATM",
                     "Canara Bank", "PNB", "IndusInd Bank"],
        "education": ["Kendriya Vidyalaya", "DPS", "National Public School",
                      "Christ University", "PES University", "RV College"],
        "healthcare": ["Apollo Clinic", "Manipal Hospital", "Fortis",
                       "MedPlus Pharmacy", "Apollo Pharmacy", "Practo Clinic"],
        "entertainment": ["PVR Cinemas", "INOX", "Gold's Gym", "Cult Fit",
                          "Playzone", "Timezone"],
        "transport": ["BMTC Bus Stop", "Metro Station", "Kempegowda Bus Terminal",
                      "Majestic", "Silk Board Junction"],
        "fuel": ["Indian Oil", "HP Petrol Bunk", "Bharat Petroleum",
                 "Shell", "Reliance Petrol"],
    }

    pois = []
    for category, weight in cat_weights.items():
        n_cat = int(n_total * weight)
        names = sample_names[category]

        for _ in range(n_cat):
            # Rejection sampling: denser near hotspots
            while True:
                lat = np.random.uniform(bbox["south"], bbox["north"])
                lon = np.random.uniform(bbox["west"], bbox["east"])
                density = _density_at(lat, lon)

                # Category-specific adjustments
                if category == "transport":
                    # Transport more spread out
                    density = density ** 0.5
                elif category in ("restaurant", "retail"):
                    # Strong commercial clustering
                    density = density ** 0.7
                elif category == "fuel":
                    # Along roads, more uniform
                    density = density ** 0.4

                if np.random.random() < density:
                    break

            # Add small jitter
            lat += np.random.normal(0, 0.001)
            lon += np.random.normal(0, 0.001)

            pois.append({
                "name": np.random.choice(names),
                "category": category,
                "lat": lat,
                "lon": lon,
            })

    df = pd.DataFrame(pois)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326",
    )
    print(f"    → {len(gdf)} POIs across {len(cat_weights)} categories")
    for cat, count in gdf.category.value_counts().items():
        print(f"      {cat}: {count}")
    return gdf


def generate_synthetic_buildings(n_total: int = 40000) -> gpd.GeoDataFrame:
    """
    Generate building centroids as population density proxy.
    Denser in the urban core, sparser toward the periphery.
    """
    print("  Generating synthetic buildings (population proxy)...")
    bbox = CITY_BBOX

    buildings = []
    for _ in range(n_total):
        while True:
            lat = np.random.uniform(bbox["south"], bbox["north"])
            lon = np.random.uniform(bbox["west"], bbox["east"])
            density = _density_at(lat, lon)
            if np.random.random() < density ** 0.6:
                break

        # Building levels: taller near center
        d = _distance_to_center(lat, lon)
        if d < 3:
            levels = np.random.choice([1, 2, 3, 4, 5, 8, 10, 12],
                                       p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05])
        elif d < 8:
            levels = np.random.choice([1, 2, 3, 4, 5],
                                       p=[0.2, 0.3, 0.25, 0.15, 0.1])
        else:
            levels = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

        buildings.append({"lat": lat, "lon": lon, "levels": int(levels)})

    df = pd.DataFrame(buildings)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lon, df.lat),
        crs="EPSG:4326",
    )
    print(f"    → {len(gdf)} buildings")
    return gdf


def generate_synthetic_roads(n_segments: int = 800) -> gpd.GeoDataFrame:
    """
    Generate synthetic road network with realistic patterns.
    Major roads radiate from center; secondary roads fill the grid.
    """
    print("  Generating synthetic road network...")
    bbox = CITY_BBOX

    roads = []
    road_types = ["primary", "secondary", "tertiary"]

    # Radial roads from center
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        pts = []
        for r in np.linspace(0, 0.12, 20):
            lat = CITY_CENTER[0] + r * np.cos(angle) + np.random.normal(0, 0.001)
            lon = CITY_CENTER[1] + r * np.sin(angle) + np.random.normal(0, 0.001)
            pts.append((lon, lat))
        roads.append({
            "name": f"Radial Road {len(roads)+1}",
            "road_type": "primary",
            "geometry": LineString(pts),
        })

    # Ring roads
    for radius in [0.03, 0.06, 0.10]:
        pts = []
        for angle in np.linspace(0, 2 * np.pi, 40):
            lat = CITY_CENTER[0] + radius * np.cos(angle) + np.random.normal(0, 0.0005)
            lon = CITY_CENTER[1] + radius * np.sin(angle) + np.random.normal(0, 0.0005)
            pts.append((lon, lat))
        pts.append(pts[0])
        roads.append({
            "name": f"Ring Road",
            "road_type": "primary" if radius > 0.05 else "secondary",
            "geometry": LineString(pts),
        })

    # Grid fill
    for _ in range(n_segments - len(roads)):
        lat1 = np.random.uniform(bbox["south"], bbox["north"])
        lon1 = np.random.uniform(bbox["west"], bbox["east"])
        direction = np.random.random() > 0.5
        length = np.random.uniform(0.005, 0.02)

        if direction:
            pts = [(lon1, lat1), (lon1 + length, lat1 + np.random.normal(0, 0.002))]
        else:
            pts = [(lon1, lat1), (lon1 + np.random.normal(0, 0.002), lat1 + length)]

        rtype = np.random.choice(road_types, p=[0.15, 0.35, 0.50])
        roads.append({
            "name": "",
            "road_type": rtype,
            "geometry": LineString(pts),
        })

    gdf = gpd.GeoDataFrame(roads, geometry="geometry", crs="EPSG:4326")
    print(f"    → {len(gdf)} road segments")
    return gdf


# ---------------------------------------------------------------------------
# Overpass API functions (for production use with live OSM data)
# ---------------------------------------------------------------------------

def _overpass_query(query: str, retries: int = 3) -> dict:
    """Execute an Overpass API query with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                raise


def fetch_pois_from_osm() -> gpd.GeoDataFrame:
    """Fetch live POIs from OpenStreetMap via Overpass API."""
    bbox = CITY_BBOX
    bbox_str = f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}"
    all_pois = []
    for category, tag_groups in POI_CATEGORIES.items():
        filters = []
        for key, values in tag_groups.items():
            for val in values:
                filters.append(f'node["{key}"="{val}"]({bbox_str});')
                filters.append(f'way["{key}"="{val}"]({bbox_str});')
        query = f'[out:json][timeout:60];({"".join(filters)});out center;'
        data = _overpass_query(query)
        for el in data.get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                all_pois.append({
                    "name": el.get("tags", {}).get("name", "Unknown"),
                    "category": category, "lat": lat, "lon": lon,
                })
        time.sleep(2)
    df = pd.DataFrame(all_pois)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ingestion(use_live_osm: bool = False):
    """
    Run full data ingestion pipeline.

    Args:
        use_live_osm: If True, fetch from Overpass API.
                      If False (default), use synthetic data.
    """
    print(f"\n{'='*60}")
    print(f"RetailIQ Data Ingestion — {CITY_NAME}")
    print(f"{'='*60}\n")

    # 1. Generate H3 grid
    print("[1/4] H3 Grid Generation")
    grid = generate_h3_grid()
    grid.to_file(DATA_DIR / "processed" / "h3_grid.geojson", driver="GeoJSON")
    print(f"  Saved: h3_grid.geojson ({len(grid)} hexagons)\n")

    # 2. POIs
    print("[2/4] POI Data")
    if use_live_osm:
        pois = fetch_pois_from_osm()
    else:
        pois = generate_synthetic_pois()
    pois.to_file(DATA_DIR / "raw" / "pois.geojson", driver="GeoJSON")
    print(f"  Saved: pois.geojson ({len(pois)} POIs)\n")

    # 3. Buildings
    print("[3/4] Building Data (population proxy)")
    buildings = generate_synthetic_buildings()
    buildings.to_file(DATA_DIR / "raw" / "buildings.geojson", driver="GeoJSON")
    print(f"  Saved: buildings.geojson ({len(buildings)} buildings)\n")

    # 4. Roads
    print("[4/4] Road Network")
    roads = generate_synthetic_roads()
    roads.to_file(DATA_DIR / "raw" / "roads.geojson", driver="GeoJSON")
    print(f"  Saved: roads.geojson ({len(roads)} road segments)\n")

    print("=" * 60)
    print("Data ingestion complete!")
    print("=" * 60)

    return grid, pois, buildings, roads


if __name__ == "__main__":
    run_ingestion()
