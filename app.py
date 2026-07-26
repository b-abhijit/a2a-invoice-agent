import os
import json
import uuid
import hashlib
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

load_dotenv()

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
BASE_URL = os.getenv("BASE_URL")
ORIGIN = os.getenv("ORIGIN")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agent.db")

missing = [name for name, value in {
    "BEARER_TOKEN": BEARER_TOKEN,
    "BASE_URL": BASE_URL,
    "ORIGIN": ORIGIN,
}.items() if not value]

if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

app = FastAPI(title="A2A Invoice Agent")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
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


def fake_ai_decide_package(package: Dict[str, Any]) -> Dict[str, Any]:
    package_id = package.get("packageId", str(uuid.uuid4()))
    text = canonical_json(package).lower()

    if "duplicate" in text or "already paid" in text:
        action = "reject_duplicate"
    elif "approval" in text or "above limit" in text or "outside authority" in text:
        action = "request_approval"
    elif "hold" in text or "verify" in text or "verification pending" in text:
        action = "hold_invoice"
    elif "conflict" in text or "mismatch" in text or "exception" in text:
        action = "open_exception"
    else:
        action = "settle_invoice"

    refs = package.get("evidenceRefs") or ["[P1]", "[P2]", "[P3]"]
    refs = refs[:3]

    facts = {
        "vendorName": package.get("vendorName", "Unknown Vendor"),
        "invoiceNumber": package.get("invoiceNumber", f"INV-{package_id[:6]}"),
        "amountMinor": int(package.get("amountMinor", 0)),
        "currency": package.get("currency", "INR")
    }

    return {
        "packageId": package_id,
        "actionId": uuid.uuid4().hex[:12],
        "action": action,
        "facts": facts,
        "evidenceRefs": refs,
        "rationale": (
            f"Chosen action is {action}. Evidence {refs[0]} and {refs[1]} support the commercial "
            f"status, and {refs[2]} is the decisive reference used to finalize the proposal."
        )
    }


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
def healthz():
    return {"ok": True}


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
            proposals = [fake_ai_decide_package(pkg) for pkg in packages]

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