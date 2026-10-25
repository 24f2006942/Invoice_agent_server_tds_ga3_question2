from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

from decisions import decide_packages, package_id_of
from storage import (
    cancel_task_atomic,
    complete_task_atomic,
    create_initial_task,
    dumps,
    get_idempotency,
    get_owned_task,
    list_owned_tasks,
    principal_for_token,
    record_replay_key,
    task_from_row,
    utc_now_iso,
)

A2A_MEDIA_TYPE = "application/a2a+json"
BATCH_MEDIA_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-receipts+json"

MAX_RESPONSE_BYTES = 512 * 1024
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
COMPLETED = "TASK_STATE_COMPLETED"
CANCELED = "TASK_STATE_CANCELED"
TERMINAL_STATES = {
    COMPLETED,
    CANCELED,
    "TASK_STATE_FAILED",
    "TASK_STATE_REJECTED",
}

app = FastAPI(
    title="A2A Invoice Action Agent",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class AgentError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
        media_type=A2A_MEDIA_TYPE,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "The request could not be completed.",
            }
        },
        media_type=A2A_MEDIA_TYPE,
    )


def json_response(payload: dict[str, Any], status_code: int = 200):
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    if len(body) > MAX_RESPONSE_BYTES:
        raise AgentError(
            413,
            "RESPONSE_TOO_LARGE",
            "The response exceeds the 512 KiB limit.",
        )

    return JSONResponse(
        content=payload,
        status_code=status_code,
        media_type=A2A_MEDIA_TYPE,
    )


def get_base_url(request: Request) -> str:
    configured = os.getenv("PUBLIC_BASE_URL", "").strip()

    if configured:
        return configured.rstrip("/")

    host = request.headers.get(
        "x-forwarded-host",
        request.headers.get("host", "localhost:8000"),
    )
    scheme = request.headers.get(
        "x-forwarded-proto",
        request.url.scheme,
    )

    return f"{scheme}://{host}".rstrip("/")


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base_url = get_base_url(request)

    card = {
        "name": "A2A Invoice Action Agent",
        "description": (
            "Reviews invoice packages and proposes safe actions "
            "using supporting evidence."
        ),
        "version": "1.0.0",
        "capabilities": {},
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [
            BATCH_MEDIA_TYPE,
            RESULTS_MEDIA_TYPE,
        ],
        "defaultOutputModes": [
            PROPOSALS_MEDIA_TYPE,
            RECEIPTS_MEDIA_TYPE,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Reviews invoice evidence and proposes safe actions."
                ),
                "tags": [
                    "invoice",
                    "evidence",
                    "approval",
                    "receipts",
                ],
            }
        ],
    }

    return JSONResponse(
        content=card,
        media_type="application/json",
    )


def authenticate(request: Request, require_content_type: bool = False):
    authorization = request.headers.get("authorization", "")
    match = re.fullmatch(r"Bearer ([^\s]+)", authorization)

    if not match:
        raise AgentError(
            401,
            "UNAUTHENTICATED",
            "A valid Bearer token is required.",
        )

    if request.headers.get("a2a-version") != "1.0":
        raise AgentError(
            400,
            "UNSUPPORTED_A2A_VERSION",
            "A2A-Version must be 1.0.",
        )

    if require_content_type:
        content_type = (
            request.headers.get("content-type", "")
            .split(";", 1)[0]
            .strip()
            .lower()
        )

        if content_type != A2A_MEDIA_TYPE:
            raise AgentError(
                415,
                "UNSUPPORTED_MEDIA_TYPE",
                f"Content-Type must be {A2A_MEDIA_TYPE}.",
            )

    return principal_for_token(match.group(1))


async def read_json_body(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise AgentError(
            400, "INVALID_JSON", "Request body must be valid JSON."
        ) from exc

    if not isinstance(body, dict):
        raise AgentError(
            400, "INVALID_REQUEST", "Request body must be a JSON object."
        )

    return body


def get_message_and_part(body: dict[str, Any]):
    message = body.get("message")

    if not isinstance(message, dict):
        raise AgentError(
            400, "INVALID_MESSAGE", "A message object is required."
        )

    if message.get("role") != "ROLE_USER":
        raise AgentError(
            400, "INVALID_ROLE", "Message role must be ROLE_USER."
        )

    message_id = message.get("messageId")

    if not isinstance(message_id, str) or not message_id.strip():
        raise AgentError(
            400, "INVALID_MESSAGE_ID", "messageId is required."
        )

    parts = message.get("parts")

    if not isinstance(parts, list) or len(parts) != 1:
        raise AgentError(
            400,
            "INVALID_PARTS",
            "The message must contain exactly one data Part.",
        )

    part = parts[0]

    if not isinstance(part, dict) or "data" not in part:
        raise AgentError(
            400, "INVALID_PART", "The message Part must contain data."
        )

    return message, part, part["data"]


def message_hash(message: dict[str, Any]) -> str:
    canonical = dumps(message)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def proposal_artifact(batch_id: str, proposals: list[dict[str, Any]]):
    return {
        "artifactId": str(uuid.uuid4()),
        "name": "Invoice action proposals",
        "parts": [
            {
                "mediaType": PROPOSALS_MEDIA_TYPE,
                "data": {
                    "batchId": batch_id,
                    "proposals": proposals,
                },
            }
        ],
    }


def receipt_artifact(batch_id: str, executions: list[dict[str, Any]]):
    return {
        "artifactId": str(uuid.uuid4()),
        "name": "Invoice action receipts",
        "parts": [
            {
                "mediaType": RECEIPTS_MEDIA_TYPE,
                "data": {
                    "batchId": batch_id,
                    "executions": executions,
                },
            }
        ],
    }


def task_state(task: dict[str, Any]) -> str:
    return str(task.get("status", {}).get("state", ""))


def validate_batch(data: Any):
    if not isinstance(data, dict):
        raise AgentError(400, "INVALID_BATCH", "Batch data must be an object.")

    batch_id = data.get("batchId")
    policy_revision = data.get("policyRevision")
    packages = data.get("packages")

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise AgentError(400, "INVALID_BATCH", "batchId is required.")

    if not isinstance(policy_revision, str) or not policy_revision.strip():
        raise AgentError(
            400, "INVALID_BATCH", "policyRevision is required."
        )

    if not isinstance(packages, list) or not packages:
        raise AgentError(
            400, "INVALID_BATCH", "packages must be a non-empty list."
        )

    if not all(isinstance(p, dict) for p in packages):
        raise AgentError(
            400, "INVALID_BATCH", "Every package must be an object."
        )

    try:
        package_ids = [package_id_of(p) for p in packages]
    except ValueError as exc:
        raise AgentError(400, "INVALID_PACKAGE", str(exc)) from exc

    if len(package_ids) != len(set(package_ids)):
        raise AgentError(
            400,
            "INVALID_PACKAGE",
            "Package IDs must be unique within the batch.",
        )

    return batch_id, packages


def get_stored_proposals(task: dict[str, Any]):
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("mediaType") == PROPOSALS_MEDIA_TYPE:
                data = part.get("data")

                if (
                    isinstance(data, dict)
                    and isinstance(data.get("proposals"), list)
                ):
                    return str(data.get("batchId", "")), data["proposals"]

    raise AgentError(
        409, "MISSING_PROPOSAL", "Stored task has no proposal artifact."
    )


def validate_results(
    message: dict[str, Any],
    task: dict[str, Any],
    batch_id: str,
    proposals: list[dict[str, Any]],
):
    if message.get("taskId") != task["id"]:
        raise AgentError(
            409, "TASK_MISMATCH", "Continuation task does not match."
        )

    if message.get("contextId") != task["contextId"]:
        raise AgentError(
            409, "CONTEXT_MISMATCH", "Continuation context does not match."
        )

    parts = message.get("parts")

    if not isinstance(parts, list) or len(parts) != 1:
        raise AgentError(
            400, "INVALID_PARTS", "Continuation must contain one data Part."
        )

    part = parts[0]

    if (
        not isinstance(part, dict)
        or part.get("mediaType") != RESULTS_MEDIA_TYPE
        or not isinstance(part.get("data"), dict)
    ):
        raise AgentError(
            400,
            "INVALID_CONTINUATION",
            "The Part must contain invoice action results.",
        )

    data = part["data"]

    if data.get("batchId") != batch_id:
        raise AgentError(
            409, "BATCH_MISMATCH", "Continuation batch does not match."
        )

    results = data.get("results")

    if not isinstance(results, list) or not results:
        raise AgentError(
            400, "INVALID_RESULTS", "results must be a non-empty list."
        )

    proposals_by_id = {
        p["packageId"]: p for p in proposals
    }

    normalized = []
    seen = set()

    for result in results:
        if not isinstance(result, dict):
            raise AgentError(
                400, "INVALID_RESULTS", "Every result must be an object."
            )

        package_id = result.get("packageId")

        if package_id not in proposals_by_id:
            raise AgentError(
                409, "PROPOSAL_MISMATCH", "Result does not match a proposal."
            )

        if package_id in seen:
            raise AgentError(
                409, "PROPOSAL_MISMATCH", "Duplicate package result."
            )

        seen.add(package_id)
        proposal = proposals_by_id[package_id]

        if (
            result.get("actionId") != proposal["actionId"]
            or result.get("action") != proposal["action"]
        ):
            raise AgentError(
                409, "PROPOSAL_MISMATCH", "Result action does not match."
            )

        outcome = result.get("outcome")

        if outcome not in {"ACCEPTED", "REJECTED"}:
            raise AgentError(
                400,
                "INVALID_OUTCOME",
                "Outcome must be ACCEPTED or REJECTED.",
            )

        nonce = result.get("receiptNonce")

        if not isinstance(nonce, str) or not nonce.strip():
            raise AgentError(
                409, "INVALID_RECEIPT", "Receipt nonce is required."
            )

        normalized.append(
            {
                "packageId": package_id,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "outcome": outcome,
                "receiptNonce": nonce,
            }
        )

    if seen != set(proposals_by_id):
        raise AgentError(
            409,
            "INCOMPLETE_RESULTS",
            "A result is required for every proposal.",
        )

    normalized.sort(key=lambda x: x["packageId"])

    executions = []

    for result in normalized:
        if result["outcome"] != "ACCEPTED":
            continue

        proposal = proposals_by_id[result["packageId"]]

        executions.append(
            {
                "packageId": proposal["packageId"],
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": result["receiptNonce"],
                "facts": copy.deepcopy(proposal["facts"]),
                "evidenceRefs": copy.deepcopy(proposal["evidenceRefs"]),
            }
        )

    signature = {
        "batchId": batch_id,
        "results": normalized,
    }

    return signature, executions


def latest_result_signature(task: dict[str, Any]):
    for message in reversed(task.get("history", [])):
        if not isinstance(message, dict):
            continue

        for part in message.get("parts", []):
            if (
                isinstance(part, dict)
                and part.get("mediaType") == RESULTS_MEDIA_TYPE
            ):
                data = part.get("data", {})
                results = data.get("results", [])

                if not isinstance(results, list):
                    continue

                copied = []

                for r in results:
                    if isinstance(r, dict):
                        copied.append(
                            {
                                "packageId": r.get("packageId"),
                                "actionId": r.get("actionId"),
                                "action": r.get("action"),
                                "outcome": r.get("outcome"),
                                "receiptNonce": r.get("receiptNonce"),
                            }
                        )

                copied.sort(key=lambda x: str(x.get("packageId", "")))

                return {
                    "batchId": data.get("batchId"),
                    "results": copied,
                }

    return None


def make_completed_task(
    task: dict[str, Any],
    message: dict[str, Any],
    executions: list[dict[str, Any]],
):
    completed = copy.deepcopy(task)
    batch_id, _ = get_stored_proposals(task)

    completed["history"].append(copy.deepcopy(message))
    completed["artifacts"].append(
        receipt_artifact(batch_id, executions)
    )
    completed["status"] = {
        "state": COMPLETED,
        "timestamp": utc_now_iso(),
    }

    return completed


message_locks: dict[str, asyncio.Lock] = {}
message_locks_guard = asyncio.Lock()


async def get_message_lock(key: str):
    async with message_locks_guard:
        if key not in message_locks:
            message_locks[key] = asyncio.Lock()

        return message_locks[key]


@app.post("/message:send")
async def message_send(request: Request):
    principal = authenticate(request, require_content_type=True)
    body = await read_json_body(request)

    message = body.get("message")

    if not isinstance(message, dict):
        raise AgentError(
            400, "INVALID_MESSAGE", "A message object is required."
        )

    message_id = message.get("messageId")

    if not isinstance(message_id, str) or not message_id.strip():
        raise AgentError(
            400, "INVALID_MESSAGE_ID", "messageId is required."
        )

    request_hash = message_hash(message)

    lock = await get_message_lock(principal + ":" + message_id)

    async with lock:
        existing = get_idempotency(principal, message_id)

        if existing:
            if existing["request_hash"] != request_hash:
                raise AgentError(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "messageId was reused with different content.",
                )

            row = get_owned_task(existing["task_id"], principal)

            if not row:
                raise AgentError(404, "NOT_FOUND", "Task not found.")

            return json_response({"task": task_from_row(row)})

        message, part, data = get_message_and_part(body)

        task_id = message.get("taskId")

        # A message with taskId is a grader result continuation.
        if task_id:
            if not isinstance(task_id, str):
                raise AgentError(
                    400, "INVALID_TASK_ID", "taskId must be a string."
                )

            row = get_owned_task(task_id, principal)

            if not row:
                raise AgentError(404, "NOT_FOUND", "Task not found.")

            task = task_from_row(row)
            batch_id, proposals = get_stored_proposals(task)

            signature, executions = validate_results(
                message, task, batch_id, proposals
            )

            state = task_state(task)

            if state == COMPLETED:
                if latest_result_signature(task) != signature:
                    raise AgentError(
                        409,
                        "TERMINAL_TASK_CONFLICT",
                        "A completed task cannot be changed.",
                    )

                status, replayed = record_replay_key(
                    principal=principal,
                    message_id=message_id,
                    request_hash=request_hash,
                    task_id=task_id,
                )

                if status == "conflict":
                    raise AgentError(
                        409,
                        "IDEMPOTENCY_CONFLICT",
                        "messageId was reused with different content.",
                    )

                return json_response({"task": replayed or task})

            if state in TERMINAL_STATES:
                raise AgentError(
                    409,
                    "TERMINAL_TASK_CONFLICT",
                    "A terminal task cannot be changed.",
                )

            if state != INPUT_REQUIRED:
                raise AgentError(
                    409,
                    "TASK_NOT_READY",
                    "Task is not waiting for results.",
                )

            completed = make_completed_task(task, message, executions)

            if len(dumps({"task": completed}).encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise AgentError(
                    413,
                    "RESPONSE_TOO_LARGE",
                    "Response exceeds the size limit.",
                )

            status, latest = complete_task_atomic(
                task_id=task_id,
                principal=principal,
                task=completed,
                message_id=message_id,
                request_hash=request_hash,
                expected_state=INPUT_REQUIRED,
            )

            if status in {"updated", "same"}:
                return json_response({"task": latest or completed})

            if status == "conflict":
                raise AgentError(
                    409,
                    "IDEMPOTENCY_CONFLICT",
                    "Continuation conflicts with an earlier request.",
                )

            if (
                latest
                and task_state(latest) == COMPLETED
                and latest_result_signature(latest) == signature
            ):
                replay_status, replayed = record_replay_key(
                    principal=principal,
                    message_id=message_id,
                    request_hash=request_hash,
                    task_id=task_id,
                )

                if replay_status in {"recorded", "same"}:
                    return json_response({"task": replayed or latest})

            raise AgentError(
                409,
                "TASK_STATE_CONFLICT",
                "Another operation changed this task first.",
            )

        # Otherwise, this is a new invoice batch.
        configuration = body.get("configuration")

        if not isinstance(configuration, dict):
            raise AgentError(
                400,
                "INVALID_CONFIGURATION",
                "Initial messages require configuration.",
            )

        accepted_modes = configuration.get("acceptedOutputModes", [])

        required_modes = {
            PROPOSALS_MEDIA_TYPE,
            RECEIPTS_MEDIA_TYPE,
        }

        if (
            not isinstance(accepted_modes, list)
            or not required_modes.issubset(
                {x for x in accepted_modes if isinstance(x, str)}
            )
        ):
            raise AgentError(
                400,
                "INVALID_CONFIGURATION",
                "Both proposals and receipts output modes are required.",
            )

        if part.get("mediaType") != BATCH_MEDIA_TYPE:
            raise AgentError(
                415,
                "UNSUPPORTED_MEDIA_TYPE",
                "Initial message must contain an invoice batch.",
            )

        batch_id, packages = validate_batch(data)

        try:
            proposals = await decide_packages(
                packages,
                policy_revision=data["policyRevision"],
            )
        except ValueError as exc:
            raise AgentError(
                502,
                "AI_DECISION_INVALID",
                "AI decisions failed evidence or schema validation.",
            ) from exc
        except RuntimeError as exc:
            if "AI is not configured" in str(exc):
                raise AgentError(
                    503,
                    "AI_NOT_CONFIGURED",
                    "Configure AI_API_KEY before submitting invoices.",
                ) from exc

            raise AgentError(
                502,
                "AI_DECISION_FAILED",
                "AI could not return a valid decision batch.",
            ) from exc
        except Exception as exc:
            raise AgentError(
                502,
                "AI_DECISION_FAILED",
                "AI could not process this invoice batch.",
            ) from exc

        context_id = message.get("contextId")

        if not isinstance(context_id, str) or not context_id.strip():
            context_id = str(uuid.uuid4())

        task = {
            "id": str(uuid.uuid4()),
            "contextId": context_id,
            "status": {
                "state": INPUT_REQUIRED,
                "timestamp": utc_now_iso(),
            },
            "artifacts": [
                proposal_artifact(batch_id, proposals)
            ],
            "history": [copy.deepcopy(message)],
        }

        if len(dumps({"task": task}).encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise AgentError(
                413,
                "RESPONSE_TOO_LARGE",
                "Response exceeds the size limit.",
            )

        try:
            status, saved = create_initial_task(
                task=task,
                principal=principal,
                message_id=message_id,
                request_hash=request_hash,
                batch_id=batch_id,
            )
        except Exception as exc:
            existing = get_idempotency(principal, message_id)

            if existing and existing["request_hash"] == request_hash:
                row = get_owned_task(existing["task_id"], principal)

                if row:
                    return json_response({"task": task_from_row(row)})

            raise AgentError(
                500,
                "TASK_PERSISTENCE_FAILED",
                "Task could not be saved safely.",
            ) from exc

        if status == "conflict":
            raise AgentError(
                409,
                "IDEMPOTENCY_CONFLICT",
                "messageId was reused with different content.",
            )

        if status == "same":
            return json_response({"task": saved})

        return json_response({"task": task})


@app.get("/tasks")
async def list_tasks(request: Request):
    principal = authenticate(request)
    return json_response({"tasks": list_owned_tasks(principal)})


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal = authenticate(request)
    row = get_owned_task(task_id, principal)

    if not row:
        raise AgentError(404, "NOT_FOUND", "Task not found.")

    return json_response(task_from_row(row))


@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = authenticate(request)
    row = get_owned_task(task_id, principal)

    if not row:
        raise AgentError(404, "NOT_FOUND", "Task not found.")

    task = task_from_row(row)
    state = task_state(task)

    if state in TERMINAL_STATES:
        raise AgentError(
            409,
            "TASK_ALREADY_TERMINAL",
            "A terminal task cannot be canceled.",
        )

    canceled = copy.deepcopy(task)
    canceled["status"] = {
        "state": CANCELED,
        "timestamp": utc_now_iso(),
    }

    status, latest = cancel_task_atomic(
        task_id=task_id,
        principal=principal,
        canceled_task=canceled,
        expected_state=state,
    )

    if status == "canceled":
        return json_response(canceled)

    if status == "not_found":
        raise AgentError(404, "NOT_FOUND", "Task not found.")

    raise AgentError(
        409,
        "TASK_STATE_CONFLICT",
        "Another operation changed this task first.",
    )