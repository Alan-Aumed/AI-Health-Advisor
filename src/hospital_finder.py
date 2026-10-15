# src/hospital_finder.py
import requests
import streamlit as st
from typing import List, Dict, Optional, Tuple

API_KEY = st.secrets["google"]["YOUR_API_KEY"]
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


def find_hospitals_by_coords(lat: float, lon: float, radius: int = 5000) -> Tuple[List[Dict], Dict]:
    """Return a tuple (hospitals_list, raw_api_response).

    The raw response is returned for diagnostics so callers can show the API status
    and error messages when no results are available.
    """
    params = {
        "key": API_KEY,
        "location": f"{lat},{lon}",
        "radius": radius,
        "type": "hospital",
    }

    try:
        resp = requests.get(PLACES_NEARBY_URL, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        # Return empty list and a simple error-like dict for diagnostics
        return [], {"status": "REQUEST_EXCEPTION", "error_message": str(e)}

    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        # Return the raw data so the caller can show status / error_message
        return [], data

    hospitals: List[Dict] = []
    for place in data.get("results", []):
        hospitals.append(
            {
                "name": place.get("name"),
                "address": place.get("vicinity") or place.get("formatted_address"),
                "rating": place.get("rating"),
                "user_ratings_total": place.get("user_ratings_total"),
                "location": place["geometry"]["location"],
                "place_id": place.get("place_id"),
                "open_now": place.get("opening_hours", {}).get("open_now"),
            }
        )

    return hospitals, data