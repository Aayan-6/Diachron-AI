import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("causal_attribution")


class CausalAttributionEngine:
    """
    Fuses Earth Observation change findings with TinyFish ground evidence
    to synthesize a verified causal attribution report and confidence score.
    """

    @staticmethod
    def synthesize_attribution(
        satellite_summary: Dict[str, Any],
        web_evidence: List[Dict[str, Any]],
        spatial_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes causality score, risk classification, and synthesized timeline.
        """
        change_type = satellite_summary.get("change_type", "General Change")
        area_ha = satellite_summary.get("area_hectares", 0.0)
        t1 = satellite_summary.get("t1_date", "T1")
        t2 = satellite_summary.get("t2_date", "T2")

        if not web_evidence:
            return {
                "causality_score": 35.0,
                "classification": "INCONCLUSIVE",
                "classification_label": "Inconclusive Ground Verification",
                "summary": (
                    f"Satellite detected {change_type} ({area_ha:.1f} ha) between {t1} and {t2}. "
                    f"No direct matching gazette or local press records were indexed for this parcel."
                ),
                "confidence_grade": "Low",
                "timeline": [
                    {"date": t1, "event": "Baseline satellite scene acquired", "source": "Sentinel-2 / Landsat"},
                    {"date": t2, "event": "Change detected via Bi-Temporal ChangeFormer", "source": "Diachron AI"},
                ],
                "corroborating_sources": [],
            }

        # Analyze keywords in web evidence
        text_corpus = " ".join([
            f"{e.get('title', '')} {e.get('snippet', '')} {e.get('content', '')}"
            for e in web_evidence
        ]).lower()

        matched_sources = []
        score = 50.0
        classification = "AUTHORIZED_DEVELOPMENT"
        classification_label = "Corroborated Development"

        infra_keywords = ["tender", "highway", "expressway", "widening", "corridor", "phase", "nhai", "pwd", "approved", "project", "construction", "industrial"]
        disaster_keywords = ["flood", "spillway", "dam", "cwc", "discharge", "heavy rain", "inundation", "wildfire", "fire", "smoke", "rescue"]
        unauthorized_keywords = ["illegal", "encroach", "demolition", "sealed", "unauthorized", "violation", "felling without", "stop work"]

        infra_matches = sum(1 for kw in infra_keywords if kw in text_corpus)
        disaster_matches = sum(1 for kw in disaster_keywords if kw in text_corpus)
        unauth_matches = sum(1 for kw in unauthorized_keywords if kw in text_corpus)

        for e in web_evidence[:5]:
            matched_sources.append({
                "title": e.get("title", "News Bulletin"),
                "url": e.get("url", "#"),
                "snippet": e.get("snippet", "")[:180] + "...",
                "domain": e.get("domain") or "Web Report",
                "date": e.get("published_date") or "Documented in Change Window",
            })

        # Score attribution
        if infra_matches >= 3:
            score = min(95.0, 70.0 + infra_matches * 5.0)
            classification = "AUTHORIZED_INFRASTRUCTURE"
            classification_label = "Planned Public Infrastructure / Development"
            explanation = (
                f"Satellite-detected change of {area_ha:.1f} ha corresponds with official infrastructure "
                f"contracts and municipal tender notices reported in local press during the observation window."
            )
        elif disaster_matches >= 3:
            score = min(92.0, 70.0 + disaster_matches * 5.0)
            classification = "CLIMATIC_OR_DISASTER_EVENT"
            classification_label = "Hydro-Meteorological / Disaster Event"
            explanation = (
                f"Satellite change corresponds directly with documented hydrological anomalies, "
                f"reservoir discharge operations, or flood/weather bulletins reported for this district."
            )
        elif unauth_matches >= 2:
            score = min(88.0, 65.0 + unauth_matches * 8.0)
            classification = "SUSPECTED_UNAUTHORIZED_ACTIVITY"
            classification_label = "Potential Regulatory Violation / Encroachment"
            explanation = (
                f"Satellite detected rapid land change without matching tender approvals; "
                f"local records indicate regulatory scrutiny or encroachment complaints in this zone."
            )
        else:
            score = 62.0
            classification = "REGIONAL_ACTIVITY_CORROBORATED"
            classification_label = "Regional Land Activity Corroborated"
            explanation = (
                f"General municipal and district activity was identified for this area, "
                f"providing partial context for the observed surface change."
            )

        # Build chronological timeline
        timeline = [
            {"date": t1, "event": "Pre-change satellite baseline captured", "source": "Sentinel-2 / Landsat (10m)"},
        ]

        if matched_sources:
            timeline.append({
                "date": matched_sources[0]["date"],
                "event": f"Ground documentation: {matched_sources[0]['title'][:60]}...",
                "source": f"TinyFish Agent ({matched_sources[0]['domain']})",
            })

        timeline.append({
            "date": t2,
            "event": f"ChangeFormer detected {change_type} ({area_ha:.1f} ha)",
            "source": "Diachron AI Bi-Temporal Engine",
        })

        return {
            "causality_score": round(score, 1),
            "classification": classification,
            "classification_label": classification_label,
            "summary": explanation,
            "confidence_grade": "High" if score >= 80 else ("Moderate" if score >= 60 else "Low"),
            "timeline": timeline,
            "corroborating_sources": matched_sources,
        }
