"""
RetailIQ Configuration
=====================
City bounds, H3 resolution, and feature parameters.
"""

# Target city: Bangalore (Bengaluru)
CITY_NAME = "Bangalore"
CITY_BBOX = {
    "south": 12.85,
    "north": 13.10,
    "west": 77.45,
    "east": 77.75,
}
CITY_CENTER = (12.9716, 77.5946)

# H3 spatial resolution
H3_RESOLUTION = 8  # ~460m edge length, matches Kontur

# Catchment ring radii (in km) for catchment scoring
CATCHMENT_RINGS_KM = [0.5, 1.0, 2.0, 3.0, 5.0]

# POI categories for feature engineering (OSM tags)
POI_CATEGORIES = {
    "restaurant": {"amenity": ["restaurant", "fast_food", "cafe", "food_court"]},
    "retail": {"shop": ["supermarket", "convenience", "mall", "department_store", "clothes", "shoes", "electronics"]},
    "bank_atm": {"amenity": ["bank", "atm"]},
    "education": {"amenity": ["school", "college", "university"]},
    "healthcare": {"amenity": ["hospital", "clinic", "pharmacy", "doctors"]},
    "entertainment": {"amenity": ["cinema", "theatre"], "leisure": ["fitness_centre", "sports_centre"]},
    "transport": {"amenity": ["bus_station"], "railway": ["station", "halt"]},
    "fuel": {"amenity": ["fuel"]},
}

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Brand categories for competitive analysis
COMPETITOR_BRANDS = {
    "optical": ["lenskart", "titan eye", "vision express", "specsmakers"],
    "qsr": ["mcdonald", "domino", "subway", "burger king", "kfc", "pizza hut"],
    "cafe": ["starbucks", "cafe coffee day", "third wave", "blue tokai"],
    "fashion": ["zara", "h&m", "pantaloons", "westside", "lifestyle"],
    "grocery": ["dmart", "more", "reliance", "big bazaar", "star bazaar", "spar"],
}

# Model parameters
DEMAND_FEATURES = [
    "pop_density_proxy",
    "poi_restaurant_count",
    "poi_retail_count",
    "poi_bank_atm_count",
    "poi_education_count",
    "poi_healthcare_count",
    "poi_transport_count",
    "catchment_pop_1km",
    "catchment_pop_3km",
    "road_density",
    "competition_index",
]

# Site score weights (sum to 1.0)
SCORE_WEIGHTS = {
    "demand_score": 0.30,
    "catchment_score": 0.25,
    "accessibility_score": 0.15,
    "competition_score": 0.15,  # inverse — less competition = higher score
    "poi_diversity_score": 0.15,
}
