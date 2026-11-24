from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TicketLog:
    id: Optional[int]
    helpscout_id: str
    subject: str
    customer_name: str
    classification: str  # support, marketing, spam
    status: str  # processed, error
    response_posted: bool
    created_at: datetime
    error_message: Optional[str] = None
    trigger: str = "webhook"
    confidence_score: Optional[int] = None
    confidence_status: Optional[str] = None
    confidence_attempts: int = 1
    confidence_summary: Optional[str] = None
    confidence_trace: Optional[str] = None
    confidence_provider: Optional[str] = None
    draft_response_html: Optional[str] = None
    evidence_summary: Optional[str] = None
    delivery_mode: Optional[str] = None
    investigator_provider: Optional[str] = None
    evaluator_provider: Optional[str] = None
    decision_note: Optional[str] = None
    # These are serialized in SQLite but represented here as structured values.
    investigation_steps: Optional[list[str]] = None
    risk_flags: Optional[list[str]] = None
    sources: Optional[list[str]] = None
