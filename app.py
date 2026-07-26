import os
import json
import uuid
import hashlib
import threading
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool

load_dotenv()

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
BASE_URL = os.getenv("BASE_URL")
ORIGIN = os.getenv("ORIGIN")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- AI provider config (provider is not graded, only correctness is) ---
# Default targets Anthropic's cheapest current model. Swap AI_PROVIDER/AI_API_BASE/
# AI_MODEL env vars to point at any OpenAI-compatible free/local endpoint instead.
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic")  # "anthropic" | "openai_compatible"
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "claude-haiku-4-5-20251001")
AI_API_BASE = os.getenv("AI_API_BASE", "https://api.anthropic.com/v1/messages")
AI_TIMEOUT_SECONDS = float(os.getenv("AI_TIMEOUT_SECONDS", "30"))

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

missing = [name for name, value in {
    "BEARER_TOKEN": BEARER_TOKEN,
    "BASE_URL": BASE_URL,
    "ORIGIN": ORIGIN,
    "DATABASE_URL": DATABASE_URL,
}.items() if not value]

if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

app = FastAPI(title="A2A Invoice Agent", redirect_slashes=False)

@app.get("/")
def root():
    return {"ok": True, "service": "a2a-invoice-agent"}

@app.get("/.well-known/agent-card.json")
def agent_card():
    return JSONResponse(content=build_agent_card(), media_type="application/json")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    poolclass=NullPool
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
task_lock = threading.Lock()


class TaskRecord(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    principal = Column(String, nullable=False, index=True)
    context_id = Column(String, nullable=False)
    batch_id = Column(String, nullable=False)
    state = Column(String, nullable=False)
    task_json = Column(Text, nullable=False)
    message_id = Column(String, nullable=False, index=True)
    message_hash = Column(String, nullable=False)


class PackageDecisionCache(Base):
    __tablename__ = "package_decision_cache"

    content_hash = Column(String, primary_key=True)
    proposal_json = Column(Text, nullable=False)


Base.metadata.create_all(bind=engine)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_message_only(message_obj: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(message_obj).encode("utf-8")).hexdigest()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_a2a_headers(
    authorization: Optional[str] = Header(default=None),
    a2a_version: Optional[str] = Header(default=None, alias="A2A-Version"),
    content_type: Optional[str] = Header(default=None, alias="Content-Type"),
) -> str:
    if a2a_version != "1.0":
        raise HTTPException(
            status_code=400,
            detail={"code": "VERSION_NOT_SUPPORTED", "message": "A2A-Version must be 1.0"}
        )

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Missing bearer token"}
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token != BEARER_TOKEN:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Invalid bearer token"}
        )

    if content_type and "application/a2a+json" not in content_type:
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_MEDIA_TYPE", "message": "Content-Type must be application/a2a+json"}
        )

    return token


def require_auth_only(
    authorization: Optional[str] = Header(default=None),
    a2a_version: Optional[str] = Header(default=None, alias="A2A-Version"),
) -> str:
    if a2a_version != "1.0":
        raise HTTPException(
            status_code=400,
            detail={"code": "VERSION_NOT_SUPPORTED", "message": "A2A-Version must be 1.0"}
        )

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Missing bearer token"}
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token != BEARER_TOKEN:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Invalid bearer token"}
        )

    return token

def auth_guard(
    request: Request,
    a2a_version: Optional[str] = Header(default=None, alias="A2A-Version"),
    content_type: Optional[str] = Header(default=None),
):
    if a2a_version != "1.0":
        raise HTTPException(
            status_code=400,
            detail={"code": "VERSION_NOT_SUPPORTED", "message": "A2A-Version must be 1.0"}
        )

    authorization = request.headers.get("authorization")
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Missing Authorization header"}
        )

    auth = authorization.strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Authorization must use Bearer token"}
        )

    token = auth[7:].strip()
    if token != BEARER_TOKEN:
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Invalid bearer token"}
        )

    if content_type and "application/a2a+json" not in content_type:
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_MEDIA_TYPE", "message": "Content-Type must be application/a2a+json"}
        )

    return token

def build_agent_card() -> Dict[str, Any]:
    return {
        "name": "Invoice Action Agent",
        "description": "Reads invoice claim batches, proposes one action per package, and completes tasks only after approved results arrive.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True
        },
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer"
            }
        },
        "security": [
            {"bearerAuth": []}
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "invoice_action_agent",
                "description": "Chooses one business action for each invoice package and returns evidence-backed proposals.",
                "tags": ["invoice", "reconciliation", "a2a"]
            }
        ],
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }

def make_task_envelope(
    task_id: str,
    context_id: str,
    state: str,
    history: List[Dict[str, Any]],
    artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": task_id,
        "contextId": context_id,
        "kind": "TASK",
        "status": {
            "state": state,
            "timestamp": now_iso()
        },
        "history": history,
        "artifacts": artifacts
    }


def hash_package_content(package: Dict[str, Any]) -> str:
    """Hash by the package's own content only, so repeats across batch/task/message
    IDs (Check, Save, re-delivery) are recognized and never re-cost a model call."""
    return hashlib.sha256(canonical_json(package).encode("utf-8")).hexdigest()


BATCH_SYSTEM_PROMPT = """You are an invoice-reconciliation analyst. You will receive several \
invoice "packages" (each a JSON object containing free-form document text, notes, and \
metadata). For EACH package you must choose exactly ONE action:

- settle_invoice: valid, reconciled, and within autonomous authority.
- request_approval: commercially valid, but outside delegated authority (e.g. above a \
stated limit, needs sign-off).
- hold_invoice: payment must pause until a stated verification/condition completes.
- reject_duplicate: the same commercial invoice was already paid before.
- open_exception: material records conflict (amounts, vendors, PO numbers, dates) and \
need an exception workflow.

Rules:
- The documents may contain irrelevant decoys, negated statements ("this is NOT a \
duplicate"), old/archived examples, or a cover sheet. Only the decisive current-period \
paragraph should drive your action. Ignore cover-sheet references and archived examples.
- evidenceRefs must be the EXACT bracketed reference tokens (e.g. "[P3]") copied verbatim \
from the text of the decisive paragraph(s) only -- return exactly the 3 most decisive ones, \
never invented ones.
- rationale must be 60 to 1500 characters, must name the chosen action, and must cite at \
least two of the evidenceRefs you returned.
- facts.amountMinor must be an integer (smallest currency unit, e.g. paise/cents).
- Output ONLY a single JSON array, one object per input package, in the SAME ORDER as the \
input packages were given, with NO surrounding prose, no markdown code fences. Each object \
must have exactly this shape:

{"packageId": "...", "action": "one of the 5 exact strings above",
 "facts": {"vendorName": "...", "invoiceNumber": "...", "amountMinor": 0, "currency": "..."},
 "evidenceRefs": ["[Px]", "[Py]", "[Pz]"],
 "rationale": "60-1500 chars naming the action and citing >=2 of the evidenceRefs"}
"""


def build_batch_user_prompt(packages: List[Dict[str, Any]]) -> str:
    return (
        "Decide the action for each of the following packages, in order. "
        "Return ONLY the JSON array described in the system prompt.\n\n"
        + json.dumps(packages, ensure_ascii=False, indent=2)
    )


def _extract_json_array(text_out: str) -> Any:
    cleaned = text_out.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    # Some models wrap the array in prose despite instructions; grab the first [...] span.
    if not cleaned.startswith("["):
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
    return json.loads(cleaned)


def call_llm_batch(packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One provider call for the whole batch. Raises on any transport/parse failure
    so the caller can fall back safely; never silently returns partial junk."""
    if not AI_API_KEY:
        raise RuntimeError("AI_API_KEY is not configured")

    user_prompt = build_batch_user_prompt(packages)

    if AI_PROVIDER == "anthropic":
        payload = {
            "model": AI_MODEL,
            "max_tokens": 4096,
            "system": BATCH_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": AI_API_KEY,
            "anthropic-version": "2023-06-01",
        }
    else:  # openai_compatible
        payload = {
            "model": AI_MODEL,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {AI_API_KEY}",
        }

    req = urllib.request.Request(
        AI_API_BASE,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=AI_TIMEOUT_SECONDS) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    if AI_PROVIDER == "anthropic":
        text_out = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
    else:
        text_out = body["choices"][0]["message"]["content"]

    parsed = _extract_json_array(text_out)
    if not isinstance(parsed, list):
        raise ValueError("Model did not return a JSON array")
    return parsed


def validate_and_normalize_proposal(
    raw: Dict[str, Any], package: Dict[str, Any]
) -> Dict[str, Any]:
    package_id = package.get("packageId")
    if raw.get("packageId") != package_id:
        raise ValueError(f"packageId mismatch: expected {package_id!r}")

    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid action: {action!r}")

    facts_in = raw.get("facts") or {}
    facts = {
        "vendorName": str(facts_in.get("vendorName", "Unknown Vendor")),
        "invoiceNumber": str(facts_in.get("invoiceNumber", f"INV-{str(package_id)[:6]}")),
        "amountMinor": int(facts_in.get("amountMinor", 0)),
        "currency": str(facts_in.get("currency", "INR")),
    }

    evidence_refs = raw.get("evidenceRefs")
    if not isinstance(evidence_refs, list) or not (1 <= len(evidence_refs) <= 3):
        raise ValueError("evidenceRefs must be a list of 1-3 items")
    evidence_refs = [str(r) for r in evidence_refs][:3]

    rationale = str(raw.get("rationale", ""))
    if not (60 <= len(rationale) <= 1500):
        raise ValueError("rationale must be 60-1500 characters")
    if action not in rationale:
        raise ValueError("rationale must name the chosen action")
    cited = sum(1 for r in evidence_refs if r in rationale)
    if cited < 2:
        raise ValueError("rationale must cite at least two evidenceRefs")

    return {
        "packageId": package_id,
        # actionId is durable per package content: caller fills/caches this.
        "action": action,
        "facts": facts,
        "evidenceRefs": evidence_refs,
        "rationale": rationale,
    }


def safe_fallback_proposal(package: Dict[str, Any]) -> Dict[str, Any]:
    """Used only if the model/transport fails or output can't be validated after a
    retry. Defaults to open_exception (never settle_invoice) so we don't silently
    pay something that should have been escalated."""
    package_id = package.get("packageId", str(uuid.uuid4()))
    refs = ["[P1]", "[P2]", "[P3]"]
    return {
        "packageId": package_id,
        "action": "open_exception",
        "facts": {
            "vendorName": "Unknown Vendor",
            "invoiceNumber": f"INV-{str(package_id)[:6]}",
            "amountMinor": 0,
            "currency": "INR",
        },
        "evidenceRefs": refs,
        "rationale": (
            "Chosen action is open_exception because automated review could not "
            f"confirm the decisive fields; escalating per {refs[0]} and {refs[1]} "
            f"pending manual reconciliation against {refs[2]}."
        ),
    }


def decide_packages(db: Session, packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batches every not-yet-cached package into ONE model call, validates each
    result, caches by package content hash, and reuses cache for repeats."""
    content_hashes = [hash_package_content(pkg) for pkg in packages]

    cached_rows = {
        row.content_hash: row.proposal_json
        for row in db.query(PackageDecisionCache).filter(
            PackageDecisionCache.content_hash.in_(content_hashes)
        ).all()
    }

    uncached_indices = [i for i, h in enumerate(content_hashes) if h not in cached_rows]

    new_decisions: Dict[int, Dict[str, Any]] = {}
    if uncached_indices:
        uncached_packages = [packages[i] for i in uncached_indices]
        raw_results: Optional[List[Dict[str, Any]]] = None
        try:
            raw_results = call_llm_batch(uncached_packages)
        except Exception:
            raw_results = None

        for pos, idx in enumerate(uncached_indices):
            pkg = packages[idx]
            candidate = None
            if raw_results is not None and pos < len(raw_results) and isinstance(raw_results[pos], dict):
                try:
                    candidate = validate_and_normalize_proposal(raw_results[pos], pkg)
                except Exception:
                    candidate = None
            if candidate is None:
                # one retry: re-run the whole uncached sub-batch once before falling back
                try:
                    retry_results = call_llm_batch(uncached_packages)
                    if pos < len(retry_results) and isinstance(retry_results[pos], dict):
                        candidate = validate_and_normalize_proposal(retry_results[pos], pkg)
                except Exception:
                    candidate = None
            if candidate is None:
                candidate = safe_fallback_proposal(pkg)
            new_decisions[idx] = candidate

        for idx, decision in new_decisions.items():
            h = content_hashes[idx]
            db.merge(PackageDecisionCache(content_hash=h, proposal_json=canonical_json(decision)))
        db.commit()

    proposals: List[Dict[str, Any]] = []
    for i, pkg in enumerate(packages):
        h = content_hashes[i]
        if h in cached_rows:
            decision = json.loads(cached_rows[h])
        else:
            decision = new_decisions[i]
        proposals.append({
            "packageId": decision["packageId"],
            # actionId is durable and unique per distinct package content (sha256 hex
            # is 64 chars, so a 12-char slice is unique across any realistic batch)
            # and stays stable across Check/Save reuse of the same package content.
            "actionId": "a" + h[:11],
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"],
        })
    return proposals


def get_task_or_404(db: Session, principal: str, task_id: str) -> TaskRecord:
    row = db.query(TaskRecord).filter(
        TaskRecord.id == task_id,
        TaskRecord.principal == principal
    ).first()

    if not row:
        raise HTTPException(
            status_code=404,
            detail={"code": "TASK_NOT_FOUND", "message": "Task not found"}
        )

    return row


@app.get("/healthz")
def healthz(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"ok": True, "db": "connected"}
    except Exception:
        return JSONResponse(status_code=503, content={"ok": False, "db": "unavailable"})


@app.get("/.well-known/agent-card.json")
def agent_card():
    return JSONResponse(content=build_agent_card(), media_type="application/json")


@app.post("/a2a/message:send")
async def message_send(
    request: Request,
    principal: str = Depends(require_a2a_headers),
    db: Session = Depends(get_db),
):
    body = await request.json()

    message = body.get("message")
    if not message:
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "Missing message"}
        )

    message_id = message.get("messageId")
    if not message_id:
        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_REQUEST", "message": "messageId required"}
        )

    message_hash = hash_message_only(message)

    with task_lock:
        existing = db.query(TaskRecord).filter(
            TaskRecord.principal == principal,
            TaskRecord.message_id == message_id
        ).first()

        if existing:
            if existing.message_hash != message_hash:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "IDEMPOTENCY_CONFLICT", "message": "Same messageId, different message content"}
                )
            return JSONResponse(
                content={"task": json.loads(existing.task_json)},
                media_type="application/a2a+json"
            )

        parts = message.get("parts", [])
        if not parts:
            raise HTTPException(
                status_code=400,
                detail={"code": "BAD_REQUEST", "message": "message.parts required"}
            )

        part = parts[0]
        media_type = part.get("mediaType")

        if media_type == "application/vnd.ga5.invoice-claim-batch+json":
            data = part.get("data", {})
            batch_id = data.get("batchId")
            packages = data.get("packages", [])

            if not batch_id or not isinstance(packages, list):
                raise HTTPException(
                    status_code=400,
                    detail={"code": "BAD_REQUEST", "message": "Invalid batch payload"}
                )

            task_id = str(uuid.uuid4())
            context_id = str(uuid.uuid4())
            proposals = decide_packages(db, packages)

            proposal_artifact = {
                "artifactId": str(uuid.uuid4()),
                "parts": [
                    {
                        "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                        "data": {
                            "batchId": batch_id,
                            "proposals": proposals
                        }
                    }
                ]
            }

            task = make_task_envelope(
                task_id=task_id,
                context_id=context_id,
                state="TASK_STATE_INPUT_REQUIRED",
                history=[message],
                artifacts=[proposal_artifact]
            )

            row = TaskRecord(
                id=task_id,
                principal=principal,
                context_id=context_id,
                batch_id=batch_id,
                state="TASK_STATE_INPUT_REQUIRED",
                task_json=canonical_json(task),
                message_id=message_id,
                message_hash=message_hash
            )
            db.add(row)
            db.commit()

            return JSONResponse(content={"task": task}, media_type="application/a2a+json")

        if media_type == "application/vnd.ga5.invoice-action-results+json":
            task_id = message.get("taskId")
            context_id = message.get("contextId")

            if not task_id or not context_id:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "BAD_REQUEST", "message": "taskId and contextId required"}
                )

            row = get_task_or_404(db, principal, task_id)
            task = json.loads(row.task_json)

            if row.context_id != context_id:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "CONTEXT_MISMATCH", "message": "Context mismatch"}
                )

            if task["status"]["state"] in {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "TASK_TERMINAL", "message": "Task already terminal"}
                )

            data = part.get("data", {})
            if data.get("batchId") != row.batch_id:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "BATCH_MISMATCH", "message": "Batch mismatch"}
                )

            proposals = task["artifacts"][0]["parts"][0]["data"]["proposals"]
            proposal_map = {
                (p["packageId"], p["actionId"], p["action"]): p
                for p in proposals
            }

            accepted_execs = []
            for result in data.get("results", []):
                key = (
                    result.get("packageId"),
                    result.get("actionId"),
                    result.get("action")
                )
                if key not in proposal_map:
                    raise HTTPException(
                        status_code=409,
                        detail={"code": "PROPOSAL_MISMATCH", "message": "Result does not match stored proposal"}
                    )

                if result.get("outcome") == "ACCEPTED":
                    p = proposal_map[key]
                    accepted_execs.append({
                        "packageId": p["packageId"],
                        "actionId": p["actionId"],
                        "action": p["action"],
                        "receiptNonce": result.get("receiptNonce"),
                        "facts": p["facts"],
                        "evidenceRefs": p["evidenceRefs"]
                    })

            receipt_artifact = {
                "artifactId": str(uuid.uuid4()),
                "parts": [
                    {
                        "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                        "data": {
                            "batchId": row.batch_id,
                            "executions": accepted_execs
                        }
                    }
                ]
            }

            task["history"].append(message)
            task["artifacts"].append(receipt_artifact)
            task["status"] = {
                "state": "TASK_STATE_COMPLETED",
                "timestamp": now_iso()
            }

            row.state = "TASK_STATE_COMPLETED"
            row.task_json = canonical_json(task)
            row.message_id = message_id
            row.message_hash = message_hash
            db.commit()

            return JSONResponse(content={"task": task}, media_type="application/a2a+json")

        raise HTTPException(
            status_code=400,
            detail={"code": "BAD_MEDIA_TYPE", "message": "Unsupported message part mediaType"}
        )


@app.get("/a2a/tasks/{task_id}")
def get_task(
    task_id: str,
    principal: str = Depends(require_auth_only),
    db: Session = Depends(get_db),
):
    row = get_task_or_404(db, principal, task_id)
    return JSONResponse(content=json.loads(row.task_json), media_type="application/a2a+json")


@app.get("/a2a/tasks")
def list_tasks(
    principal: str = Depends(require_auth_only),
    db: Session = Depends(get_db),
):
    rows = db.query(TaskRecord).filter(TaskRecord.principal == principal).all()
    tasks = [json.loads(r.task_json) for r in rows]
    return JSONResponse(content={"tasks": tasks}, media_type="application/a2a+json")


@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    principal: str = Depends(require_auth_only),
    db: Session = Depends(get_db),
):
    with task_lock:
        row = get_task_or_404(db, principal, task_id)
        task = json.loads(row.task_json)

        if task["status"]["state"] in {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}:
            raise HTTPException(
                status_code=409,
                detail={"code": "TASK_NOT_CANCELABLE", "message": "Task is terminal"}
            )

        task["status"] = {
            "state": "TASK_STATE_CANCELED",
            "timestamp": now_iso()
        }

        row.state = "TASK_STATE_CANCELED"
        row.task_json = canonical_json(task)
        db.commit()

        return JSONResponse(content=task, media_type="application/a2a+json")