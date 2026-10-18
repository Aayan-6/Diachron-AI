/**
 * Typed API Client for Analyst Feedback & Subsequent Reranking (ISRO PS 26167 §2.2.5).
 */

import { API_BASE_URL } from './api';
import type { ChangeCandidate } from './types';

export interface RerankCandidatesRequest {
  candidates: ChangeCandidate[];
  analyst_verdicts: Record<string, any>;
  session_id?: string;
  beta_promote?: number;
  gamma_demote?: number;
}

export interface RerankCandidatesResponse {
  reranked_candidates: ChangeCandidate[];
  promoted_count: number;
  demoted_count: number;
  feedback_active: boolean;
  audit_summary: {
    session_id?: string;
    total_candidates?: number;
    confirmed_count?: number;
    rejected_count?: number;
    unreviewed_count?: number;
    promoted_count?: number;
    demoted_count?: number;
    timestamp?: string;
  };
}

export interface RefineArchiveSearchRequest {
  query: string;
  confirmed_tile_ids: string[];
  rejected_tile_ids: string[];
  limit?: number;
  session_id?: string;
}

export interface RefineArchiveSearchResponse {
  total_matches: number;
  results: any[];
  query_refined: boolean;
  confirmed_count: number;
  rejected_count: number;
  explanation: string;
}

export interface AuditEventRecord {
  event_id: string;
  timestamp: string;
  event_type: string;
  analyst_id: string;
  details: Record<string, any>;
}

export interface AuditTrailResponse {
  session_id: string;
  events: AuditEventRecord[];
  total_events: number;
  updated_at?: string;
  error?: string;
}

/**
 * Dynamically reranks change candidates based on analyst confirmations and rejections.
 */
export async function rerankCandidatesWithFeedback(
  req: RerankCandidatesRequest
): Promise<RerankCandidatesResponse> {
  const res = await fetch(`${API_BASE_URL}/feedback/rerank-candidates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Candidate reranking failed: ${err}`);
  }

  return res.json();
}

/**
 * Refines multimodal archive search using Rocchio Relevance Feedback on confirmed/rejected tiles.
 */
export async function refineArchiveSearchWithFeedback(
  req: RefineArchiveSearchRequest
): Promise<RefineArchiveSearchResponse> {
  const res = await fetch(`${API_BASE_URL}/feedback/refine-archive`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Archive search refinement failed: ${err}`);
  }

  return res.json();
}

/**
 * Logs an analyst triage or verification event to the persistent session audit trail.
 */
export async function logAuditEvent(
  sessionId: string,
  eventType: string,
  details: Record<string, any>,
  analystId = 'analyst_1'
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/feedback/audit-event`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      event_type: eventType,
      details,
      analyst_id: analystId,
    }),
  });

  if (!res.ok) {
    console.warn('Failed to log audit event');
  }
  return res.json();
}

/**
 * Retrieves the complete chronological audit trail and decision events for a session.
 */
export async function fetchAuditTrail(
  sessionId: string
): Promise<AuditTrailResponse> {
  const res = await fetch(`${API_BASE_URL}/feedback/audit-trail/${encodeURIComponent(sessionId)}`);
  if (!res.ok) {
    return { session_id: sessionId, events: [], total_events: 0 };
  }
  return res.json();
}
