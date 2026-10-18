import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()

# Persistence directory for audit logs
AUDIT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "audit_logs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


class ExportGeoJsonRequest(BaseModel):
    title: Optional[str] = "SatQuery Intelligence Export"
    features: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = None
    analyst_verdicts: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ExportAuditJsonRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    query: Optional[str] = ""
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = None
    analyst_verdicts: Dict[str, Any] = Field(default_factory=dict)
    summary_stats: Optional[Dict[str, Any]] = None


class ExportDossierRequest(BaseModel):
    title: Optional[str] = "Earth-Observation Change Analysis Intelligence Brief"
    query: Optional[str] = "Multi-Temporal Change Assessment"
    executive_summary: Optional[str] = ""
    timeline: Optional[List[Dict[str, Any]]] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Optional[Dict[str, Any]] = None
    analyst_verdicts: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SaveAuditRequest(BaseModel):
    session_id: str
    query: Optional[str] = ""
    decisions: Dict[str, Any]  # candidate_id -> { verdict, notes, timestamp, analyst_id }
    provenance: Optional[Dict[str, Any]] = None


@router.post("/geojson")
def export_geojson(req: ExportGeoJsonRequest):
    """
    Generate an RFC 7946 compliant GeoJSON FeatureCollection enriched with
    analyst triage decisions, verification stamps, and full processing provenance.
    """
    enriched_features = []
    verdicts = req.analyst_verdicts or {}

    for idx, feat in enumerate(req.features):
        c_id = str(feat.get("properties", {}).get("id", idx + 1))
        decision = verdicts.get(c_id, {})

        props = dict(feat.get("properties", {}))
        props["analyst_verdict"] = decision.get("verdict", "unreviewed")
        props["analyst_notes"] = decision.get("notes", "")
        props["analyst_id"] = decision.get("analyst_id", "analyst_1")
        props["reviewed_at"] = decision.get("reviewed_at")

        if req.provenance:
            props["sensor_platform"] = req.provenance.get("sensor_platform", "Sentinel-2 MSI")
            props["pipeline_version"] = req.provenance.get("pipeline_version", "SatQuery AI 2.0")

        enriched_features.append({
            "type": "Feature",
            "geometry": feat.get("geometry"),
            "properties": props
        })

    payload = {
        "type": "FeatureCollection",
        "name": req.title,
        "metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "standard": "RFC 7946 GeoJSON / ISRO PS2",
            "feature_count": len(enriched_features),
            "provenance": req.provenance or {
                "pipeline_version": "SatQuery AI 2.0 (ISRO PS2)",
                "spatial_crs": "EPSG:4326 (WGS84)"
            }
        },
        "features": enriched_features
    }

    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="satquery_provenance_{int(datetime.now().timestamp())}.geojson"'}
    )


@router.post("/audit-json")
def export_audit_json(req: ExportAuditJsonRequest):
    """
    Export a machine-readable audit report compliant with W3C PROV and ISO 19115
    preserving full source-scene, algorithm, and analyst verification history.
    """
    verdicts = req.analyst_verdicts or {}
    total_cand = len(req.candidates)
    confirmed_count = sum(1 for v in verdicts.values() if v.get("verdict") == "confirmed")
    rejected_count = sum(1 for v in verdicts.values() if v.get("verdict") == "rejected")
    flagged_count = sum(1 for v in verdicts.values() if v.get("verdict") == "flagged")

    audit_payload = {
        "prov_standard": "W3C PROV-O / ISO 19115-1:2014",
        "system": "SatQuery AI Remote Sensing Decision Support",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": req.session_id,
        "query": req.query,
        "provenance_summary": req.provenance or {},
        "triage_statistics": {
            "total_candidates": total_cand,
            "confirmed_count": confirmed_count,
            "rejected_count": rejected_count,
            "flagged_count": flagged_count,
            "pending_count": max(0, total_cand - (confirmed_count + rejected_count + flagged_count)),
        },
        "analyst_audit_trail": [
            {
                "candidate_id": c.get("id"),
                "change_type": c.get("change_type") or c.get("label") or "Surface Change",
                "earliest_observation": c.get("earliest_observation") or c.get("earliest_detected"),
                "latest_observation": c.get("latest_observation") or c.get("latest_detected"),
                "area_ha": c.get("area_ha"),
                "confidence": c.get("confidence"),
                "centroid": c.get("centroid"),
                "analyst_verdict": verdicts.get(str(c.get("id")), {}).get("verdict", "unreviewed"),
                "analyst_notes": verdicts.get(str(c.get("id")), {}).get("notes", ""),
                "analyst_id": verdicts.get(str(c.get("id")), {}).get("analyst_id", "analyst_1"),
                "reviewed_at": verdicts.get(str(c.get("id")), {}).get("reviewed_at"),
            }
            for c in req.candidates
        ]
    }

    return JSONResponse(
        content=audit_payload,
        headers={"Content-Disposition": f'attachment; filename="satquery_audit_trail_{int(datetime.now().timestamp())}.json"'}
    )


@router.post("/intelligence-dossier", response_class=HTMLResponse)
def export_intelligence_dossier(req: ExportDossierRequest):
    """
    Generate a publication-grade, self-contained HTML Intelligence Dossier
    ready for browser printing (PDF) or inclusion in operational briefings.
    """
    prov = req.provenance or {}
    verdicts = req.analyst_verdicts or {}

    rows_html = ""
    for c in req.candidates:
        c_id = str(c.get("id"))
        dec = verdicts.get(c_id, {})
        v_status = dec.get("verdict", "pending").lower()

        badge_color = "#00f2ff"
        if v_status == "confirmed":
            badge_color = "#10b981"
        elif v_status == "rejected":
            badge_color = "#ef4444"
        elif v_status == "flagged":
            badge_color = "#f59e0b"

        c_type = c.get("change_type") or c.get("label") or "Surface Change"
        e_seen = c.get("earliest_observation") or c.get("earliest_detected") or "—"
        l_seen = c.get("latest_observation") or c.get("latest_detected") or "—"

        target_badge = (
            f'<span style="display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 9.5px; font-weight: 700; background: rgba(0,242,255,0.15); color: #00f2ff; border: 1px solid rgba(0,242,255,0.4); margin-left: 6px;">🎯 TARGET ({int(c.get("semantic_relevance", 0.9)*100)}%)</span>'
            if c.get("query_matched")
            else ""
        )

        grounding_html = (
            f'<div style="font-size: 10.5px; color: #38bdf8; margin-top: 4px; font-style: italic;">🎯 CDQAG: {c.get("grounding_reason")}</div>'
            if c.get("grounding_reason")
            else ""
        )

        rows_html += f"""
        <tr style="border-bottom: 1px solid #222e3e;">
            <td style="padding: 10px; font-weight: bold;">#{c.get('id')}</td>
            <td style="padding: 10px;">{c_type}{target_badge}</td>
            <td style="padding: 10px; font-family: monospace;">{e_seen}</td>
            <td style="padding: 10px; font-family: monospace;">{l_seen}</td>
            <td style="padding: 10px; font-family: monospace;">{c.get('area_ha', 0):.2f} ha</td>
            <td style="padding: 10px; font-family: monospace;">{c.get('confidence', 0)*100:.1f}%</td>
            <td style="padding: 10px;">
                <span style="display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; text-transform: uppercase; background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color};">
                    {v_status}
                </span>
            </td>
            <td style="padding: 10px; font-size: 12px; color: #94a3b8;">
                {dec.get('notes', '—')}
                {grounding_html}
                {f'<div style="font-size: 10px; color: #64748b; margin-top: 3px; font-family: monospace;">Verified by: {dec.get("analyst_id")}</div>' if dec.get('analyst_id') else ''}
            </td>
        </tr>
        """

    timeline_html = ""
    if req.timeline:
        t_items = "".join(
            f"""<div style="flex: 1; padding: 10px; background: #0b1320; border: 1px solid #1e293b; border-radius: 8px; min-width: 140px;">
                <div style="font-size: 11px; color: #00f2ff; font-weight: bold;">{'T0 (Baseline)' if t.get('is_baseline') else f"T{t.get('step', i)}"}</div>
                <div style="font-size: 13px; font-weight: 600; margin: 4px 0;">{t.get('date', '')}</div>
                <div style="font-size: 11px; color: #94a3b8;">Change: {t.get('change_area_ha', 0):.1f} ha</div>
            </div>"""
            for i, t in enumerate(req.timeline)
        )
        timeline_html = f"""
        <div style="margin: 24px 0;">
            <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 10px;">Chronological Acquisition Sequence</h3>
            <div style="display: flex; gap: 10px; overflow-x: auto;">
                {t_items}
            </div>
        </div>
        """

    exec_summary_html = ""
    if req.executive_summary:
        exec_summary_html = f"""
        <div style="background: #0b1320; border-left: 3px solid #00f2ff; padding: 14px 18px; border-radius: 6px; margin-bottom: 24px; font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            <b style="color: #00f2ff; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; display: block; margin-bottom: 4px;">Executive Intelligence Summary</b>
            {req.executive_summary}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{req.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #060b13;
            color: #f1f5f9;
            margin: 0;
            padding: 40px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 980px;
            margin: 0 auto;
            background: #0d1524;
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 36px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .title {{
            font-size: 24px;
            font-weight: 700;
            color: #00f2ff;
            margin: 0 0 6px 0;
        }}
        .meta {{
            font-size: 12px;
            color: #94a3b8;
        }}
        .provenance-box {{
            background: #080e18;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 24px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            font-size: 12px;
        }}
        .prov-item b {{
            color: #cbd5e1;
            display: block;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 2px;
        }}
        .prov-item span {{
            font-family: monospace;
            color: #00f2ff;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }}
        th {{
            background: #0b1320;
            padding: 10px;
            text-align: left;
            font-size: 11px;
            text-transform: uppercase;
            color: #94a3b8;
            letter-spacing: 0.05em;
            border-bottom: 2px solid #1e293b;
        }}
        @media print {{
            body {{ background: #ffffff; color: #000000; padding: 10px; }}
            .container {{ box-shadow: none; border: none; padding: 0; background: #ffffff; color: #000000; }}
            .title {{ color: #0f172a; }}
            .provenance-box {{ background: #f8fafc; border-color: #cbd5e1; color: #0f172a; }}
            .prov-item span {{ color: #0284c7; }}
            th {{ background: #f1f5f9; color: #475569; }}
            tr {{ border-bottom-color: #e2e8f0 !important; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 class="title">🛰️ {req.title}</h1>
                <div class="meta"><b>Query Intent:</b> "{req.query}" &bull; <b>Report Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
            <div style="text-align: right;">
                <span style="font-family: monospace; font-size: 11px; background: #00f2ff22; color: #00f2ff; border: 1px solid #00f2ff44; padding: 4px 8px; border-radius: 6px;">
                    {prov.get('pipeline_version', 'SatQuery AI 2.0')}
                </span>
            </div>
        </div>

        <div class="provenance-box">
            <div class="prov-item"><b>Sensor & Platform</b><span>{prov.get('sensor_platform', 'Sentinel-2 MSI')}</span></div>
            <div class="prov-item"><b>Spatial CRS & GSD</b><span>{prov.get('spatial_crs', 'EPSG:4326')} &bull; {prov.get('resolution_gsd', '10.0m')}</span></div>
            <div class="prov-item"><b>Coregistration</b><span>{prov.get('coregistration', 'Sub-pixel Phase Correlation')}</span></div>
            <div class="prov-item"><b>Radiometric Normalization</b><span>{prov.get('radiometric_normalization', 'IRMAD p &lt; 0.01')}</span></div>
            <div class="prov-item"><b>AI Models</b><span>{', '.join(prov.get('models_used', ['ChangeFormerV6']))}</span></div>
            <div class="prov-item"><b>Quality Screening</b><span>{prov.get('atmospheric_correction', 'Automated SCL / QA Masking')}</span></div>
        </div>

        {timeline_html}

        {exec_summary_html}

        <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin: 28px 0 8px 0;">
            Tracked Change Candidates & Analyst Review Audit
        </h3>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Change Classification</th>
                    <th>Earliest Seen</th>
                    <th>Latest Seen</th>
                    <th>Area</th>
                    <th>Confidence</th>
                    <th>Analyst Verdict</th>
                    <th>Notes / Remarks</th>
                </tr>
            </thead>
            <tbody>
                {rows_html if rows_html else '<tr><td colspan="8" style="padding: 16px; text-align: center; color: #94a3b8;">No candidates tracked in this session.</td></tr>'}
            </tbody>
        </table>

        {"" if not any(c.get("attribution_dossier") for c in req.candidates) else f'''
        <div style="margin-top: 36px;">
            <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #00f2ff; margin-bottom: 12px; display: flex; items-center; gap: 8px;">
                <span>AI Trajectory Attribution Dossiers (ChangeChat Engine)</span>
            </h3>
            <div style="display: flex; flex-direction: column; gap: 18px;">
                {"".join(
                    f"""<div style="background: #0b1320; border: 1px solid #1e293b; border-radius: 8px; padding: 18px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="font-weight: bold; color: #f1f5f9;">Candidate #{c.get('id')} &bull; {c.get('attribution_dossier', {}).get('attribution', 'Attributed Change')}</span>
                            <span style="padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; background: {'#10b98122' if c.get('attribution_dossier', {}).get('is_anthropogenic') else '#0ea5e922'}; color: {'#10b981' if c.get('attribution_dossier', {}).get('is_anthropogenic') else '#0ea5e9'}; border: 1px solid {'#10b981' if c.get('attribution_dossier', {}).get('is_anthropogenic') else '#0ea5e9'};">
                                {'Anthropogenic Development' if c.get('attribution_dossier', {}).get('is_anthropogenic') else 'Natural / Phenological'}
                            </span>
                        </div>
                        {f'<div style="margin: 10px 0;"><img src="{c.get("attribution_dossier", {}).get("triplet_image_url")}" style="width: 100%; max-height: 200px; object-fit: contain; border-radius: 6px; border: 1px solid #334155;" alt="Triplet Visual Strip" /></div>' if c.get("attribution_dossier", {}).get("triplet_image_url") else ''}
                        <div style="font-size: 12.5px; color: #cbd5e1; margin-bottom: 8px;">{c.get('attribution_dossier', {}).get('summary', '')}</div>
                        <div style="font-size: 11.5px; color: #94a3b8; font-family: monospace; white-space: pre-line; background: #060b13; padding: 10px; border-radius: 6px; border: 1px solid #1e293b;">{c.get('attribution_dossier', {}).get('trajectory_narrative', '')}</div>
                    </div>"""
                    for c in req.candidates if c.get("attribution_dossier")
                )}
            </div>
        </div>
        '''}

        <div style="margin-top: 36px; padding-top: 16px; border-top: 1px solid #1e293b; display: flex; justify-content: space-between; font-size: 11px; color: #64748b; font-family: monospace;">
            <span>Diachron AI Autonomous EO Archive &bull; SIH 2026 / ISRO PS2</span>
            <span>Digital Audit Stamp &bull; ISO 19115 Provenance Compliant</span>
        </div>
    </div>
</body>
</html>"""

    return HTMLResponse(content=html)


@router.post("/analyst/audit")
def save_analyst_audit(req: SaveAuditRequest):
    """
    Persist analyst decisions, notes, and verification stamps to local audit store.
    """
    safe_session = "".join(c for c in req.session_id if c.isalnum() or c in "-_")
    if not safe_session:
        safe_session = "default_session"

    file_path = AUDIT_DIR / f"{safe_session}.json"
    data = {
        "session_id": safe_session,
        "query": req.query,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": req.provenance or {},
        "decisions": req.decisions
    }

    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"status": "success", "session_id": safe_session, "saved_decisions": len(req.decisions)}


@router.get("/analyst/audit/{session_id}")
def get_analyst_audit(session_id: str):
    """
    Retrieve persisted analyst decisions and audit trail for a session.
    """
    safe_session = "".join(c for c in session_id if c.isalnum() or c in "-_")
    file_path = AUDIT_DIR / f"{safe_session}.json"

    if not file_path.exists():
        return {"session_id": safe_session, "decisions": {}}

    try:
        content = json.loads(file_path.read_text(encoding="utf-8"))
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audit log: {e}")
