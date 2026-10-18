import logging
from typing import Dict, Any, Optional, List
import urllib.request
import json

logger = logging.getLogger("spatial_anchor")


class SpatialAnchorResolver:
    """
    Transforms raw geographic coordinates and change metadata into
    semantically rich search anchors and investigation targets.
    """

    @staticmethod
    def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
        """
        Reverse geocodes (lat, lon) using OpenStreetMap Nominatim with a timeout.
        Falls back to approximate coordinate formatting if offline.
        """
        headers = {"User-Agent": "DiachronAI-Geospatial-Sentinel/1.0"}
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=14&addressdetails=1"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                address = data.get("address", {})
                
                return {
                    "display_name": data.get("display_name", f"Location ({lat:.3f}, {lon:.3f})"),
                    "village": address.get("village") or address.get("hamlet") or address.get("suburb"),
                    "town_or_city": address.get("city") or address.get("town") or address.get("municipality"),
                    "subdistrict": address.get("county") or address.get("state_district") or address.get("tehsil"),
                    "district": address.get("state_district") or address.get("county"),
                    "state": address.get("state"),
                    "country": address.get("country", "India"),
                }
        except Exception as e:
            logger.warning(f"Reverse geocode failed or timed out: {e}")
            return {
                "display_name": f"Coordinates [{lat:.4f}, {lon:.4f}]",
                "village": None,
                "town_or_city": None,
                "subdistrict": None,
                "district": None,
                "state": None,
                "country": "India",
            }

    @staticmethod
    def generate_investigative_queries(
        spatial_context: Dict[str, Any],
        change_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Generates targeted investigation queries tailored for TinyFish Search & Web Agent.
        """
        place_tokens = []
        if spatial_context.get("village"):
            place_tokens.append(spatial_context["village"])
        if spatial_context.get("town_or_city"):
            place_tokens.append(spatial_context["town_or_city"])
        if spatial_context.get("subdistrict") and spatial_context["subdistrict"] not in place_tokens:
            place_tokens.append(spatial_context["subdistrict"])
        if spatial_context.get("district") and spatial_context["district"] not in place_tokens:
            place_tokens.append(spatial_context["district"])
        if spatial_context.get("state") and spatial_context["state"] not in place_tokens:
            place_tokens.append(spatial_context["state"])

        primary_loc = " ".join(place_tokens[:3]) if place_tokens else "India"

        # Map change types to investigative domain terms
        queries = []

        if "water" in change_type.lower() or "reservoir" in change_type.lower() or "flood" in change_type.lower():
            queries.append({
                "type": "hydrological_bulletin",
                "query": f"{primary_loc} dam reservoir water discharge flood spillway",
                "purpose": "Verify official reservoir discharge, flood alert, or drought conditions",
            })
            queries.append({
                "type": "disaster_news",
                "query": f"{primary_loc} water logging overflow breach rainfall",
                "purpose": "Find local news reports on flood damage or reservoir depletion",
            })
        elif "vegetation" in change_type.lower() or "forest" in change_type.lower() or "canopy" in change_type.lower() or "greenery" in change_type.lower():
            queries.append({
                "type": "environmental_clearance",
                "query": f"{primary_loc} forest tree cutting environmental clearance project highway",
                "purpose": "Find infrastructure clearances or tree felling permissions",
            })
            queries.append({
                "type": "wildfire_incident",
                "query": f"{primary_loc} wildfire forest fire smoke incident alert",
                "purpose": "Corroborate thermal anomaly or accidental forest burn",
            })
        elif "construction" in change_type.lower() or "urban" in change_type.lower() or "soil" in change_type.lower() or "excavation" in change_type.lower():
            queries.append({
                "type": "infrastructure_tender",
                "query": f"{primary_loc} construction tender industrial corridor highway expansion development",
                "purpose": "Identify authorized government or municipal construction projects",
            })
            queries.append({
                "type": "encroachment_news",
                "query": f"{primary_loc} land acquisition demolition encroachment notice",
                "purpose": "Check for illegal construction or municipal land enforcement",
            })
        else:
            queries.append({
                "type": "general_environmental_event",
                "query": f"{primary_loc} land use change development disaster alert",
                "purpose": "Identify significant ground events explaining satellite change",
            })

        return queries
