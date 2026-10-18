"""
Analyst Feedback & Subsequent Reranking API Routes (ISRO PS 26167 §2.2.5)
==========================================================================
Provides HTTP endpoints for:
  - Dynamic candidate reranking via Rocchio Active Learning
  - Multimodal Archive Search query refinement using confirmed/rejected exemplars
  - Tamper-evident session audit logging and event retrieval
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services import feedback_rerank_service, archive_indexer, embedding_service

router = APIRouter()


class RerankCandidatesRequest(BaseModel):
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    analyst_verdicts: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = "default_session"
    beta_promote: Optional[float] = 0.35
    gamma_demote: Optional[float] = 0.45


class RefineArchiveSearchRequest(BaseModel):
    query: str
    confirmed_tile_ids: List[str] = Field(default_factory=list)
    rejected_tile_ids: List[str] = Field(default_factory=list)
    limit: Optional[int] = 15
    session_id: Optional[str] = "default_session"


class AuditEventRequest(BaseModel):
    session_id: str
    event_type: str
    details: Dict[str, Any] = Field(default_factory=dict)
    analyst_id: Optional[str] = "analyst_1"


@router.post("/rerank-candidates")
async def rerank_candidates_endpoint(req: RerankCandidatesRequest) -> Dict[str, Any]:
    """
    Reranks the review queue candidates dynamically based on analyst confirmations,
    rejections, and diagnostic notes (§2.2.5).
    """
    try:
        result = feedback_rerank_service.rerank_candidates_with_feedback(
            candidates=req.candidates,
            analyst_verdicts=req.analyst_verdicts,
            beta_promote=req.beta_promote or 0.35,
            gamma_demote=req.gamma_demote or 0.45,
            session_id=req.session_id or "default_session",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {e}")


@router.post("/refine-archive")
async def refine_archive_search_endpoint(req: RefineArchiveSearchRequest) -> Dict[str, Any]:
    """
    Refines archive search using Rocchio Relevance Feedback on confirmed (relevant)
    and rejected (irrelevant) satellite scene exemplars (§2.2.1 & §2.2.5).
    """
    try:
        indexer = archive_indexer.get_archive_indexer()
        if not indexer.is_indexed():
            indexer.scan_and_index()

        if len(indexer.tiles) == 0:
            return {"total_matches": 0, "results": [], "query_refined": False}

        # 1. Base query embedding
        q_vec = embedding_service.embed_text_query(req.query or "")

        # 2. Extract vectors for confirmed and rejected tiles
        pos_vecs = []
        neg_vecs = []
        tile_lookup = {t.tile_id: t for t in indexer.tiles}

        for tid in req.confirmed_tile_ids:
            if tid in tile_lookup:
                pos_vecs.append(tile_lookup[tid].embedding)

        for tid in req.rejected_tile_ids:
            if tid in tile_lookup:
                neg_vecs.append(tile_lookup[tid].embedding)

        # 3. Apply Rocchio query vector shift
        refined_q = feedback_rerank_service.refine_query_vector_rocchio(
            query_vector=q_vec,
            relevant_vectors=pos_vecs,
            irrelevant_vectors=neg_vecs,
        )

        # 4. Search with refined vector
        similarities = embedding_service.batch_cosine_similarity(refined_q, indexer.embedding_matrix)

        matched = []
        for idx, tile in enumerate(indexer.tiles):
            score = float(similarities[idx])
            rec = tile.to_dict()
            rec["similarity_score"] = round(score, 4)
            rec["is_feedback_confirmed"] = tile.tile_id in req.confirmed_tile_ids
            rec["is_feedback_rejected"] = tile.tile_id in req.rejected_tile_ids
            matched.append(rec)

        matched.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = matched[:req.limit]

        # 5. Log audit event
        feedback_rerank_service.append_audit_event(
            session_id=req.session_id or "default_session",
            event_type="ARCHIVE_SEARCH_REFINED",
            details={
                "original_query": req.query,
                "confirmed_count": len(pos_vecs),
                "rejected_count": len(neg_vecs),
                "top_result_id": results[0]["tile_id"] if results else None,
            },
        )

        return {
            "total_matches": len(matched),
            "results": results,
            "query_refined": True,
            "confirmed_count": len(pos_vecs),
            "rejected_count": len(neg_vecs),
            "explanation": (
                f"Rocchio query refinement applied with {len(pos_vecs)} confirmed and "
                f"{len(neg_vecs)} rejected exemplar scenes."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Archive refinement failed: {e}")


@router.post("/audit-event")
async def log_audit_event_endpoint(req: AuditEventRequest) -> Dict[str, Any]:
    """
    Logs an immutable analyst triage or feedback decision event.
    """
    try:
        event = feedback_rerank_service.append_audit_event(
            session_id=req.session_id,
            event_type=req.event_type,
            details=req.details,
            analyst_id=req.analyst_id or "analyst_1",
        )
        return {"status": "success", "event": event}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit logging failed: {e}")


@router.get("/audit-trail/{session_id}")
async def get_audit_trail_endpoint(session_id: str) -> Dict[str, Any]:
    """
    Retrieves the chronological audit trail and verification records for a session.
    """
    try:
        trail = feedback_rerank_service.get_session_audit_trail(session_id)
        return trail
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit retrieval failed: {e}")
