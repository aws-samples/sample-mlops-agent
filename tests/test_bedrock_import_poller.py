"""Unit tests for the Bedrock Custom Model Import poller (R5).

Bedrock does NOT emit an EventBridge state-change event for model-import
jobs, so we run the poller on a 15-min schedule. These tests pin its
DDB-scan + terminal-state branching contract:

  * scans only kind='bedrock_import' rows whose job is IN_PROGRESS
  * skips in-flight (InProgress / Submitted / Queued) Bedrock statuses
  * on terminal status (Completed / Failed / Stopped):
      - updates DDB jobs.<id>.status, completed_at, updated_at
      - resumes the agent thread via _invoke_agentcore
  * unknown statuses are treated as in-flight (no DDB writes, no resume)
"""
import importlib
import os
import sys
from unittest.mock import MagicMock


def _import_poller():
    """Import lambda/callback/poller.py with handler.py as its sibling.

    poller.py does `from handler import ...`, so we put callback/ on
    sys.path and import handler first to seed sys.modules.
    """
    cb_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "callback")
    )
    for p in list(sys.path):
        if "/lambda/callback" in p or "/lambda/skills/" in p:
            sys.path.remove(p)
    sys.path.insert(0, cb_dir)
    sys.modules.pop("handler", None)
    sys.modules.pop("poller", None)
    importlib.import_module("handler")
    return importlib.import_module("poller")


# ---------- _scan_in_flight_imports ----------

def test_scan_filters_to_bedrock_import_in_progress_rows():
    """Scan must drop rows whose nested jobs.<id> is wrong kind or not IN_PROGRESS."""
    p = _import_poller()

    fake_table = MagicMock()
    fake_table.scan.return_value = {
        "Items": [
            {
                "task_id": "thr-keep",
                "user_id": "u1",
                "jobs": {
                    "jb1": {"kind": "bedrock_import", "status": "IN_PROGRESS"},
                    "jb2": {"kind": "training",       "status": "IN_PROGRESS"},   # wrong kind
                    "jb3": {"kind": "bedrock_import", "status": "COMPLETED"},      # already closed
                },
            },
            {  # row missing thread_id — must be skipped
                "user_id": "u-x",
                "jobs": {"jbX": {"kind": "bedrock_import", "status": "IN_PROGRESS"}},
            },
            {  # row with non-dict jobs — must be skipped without crashing
                "task_id": "thr-bad",
                "jobs": "not-a-map",
            },
        ],
    }
    p.boto3.resource = lambda *_a, **_kw: MagicMock(Table=lambda _n: fake_table)  # type: ignore[attr-defined]

    matches = p._scan_in_flight_imports()
    assert len(matches) == 1
    m = matches[0]
    assert m["thread_id"] == "thr-keep"
    assert m["job_id"] == "jb1"
    assert m["user_id"] == "u1"


def test_scan_paginates_via_last_evaluated_key():
    """Scan must follow LastEvaluatedKey until exhausted."""
    p = _import_poller()
    fake_table = MagicMock()
    fake_table.scan.side_effect = [
        {
            "Items": [{
                "task_id": "thr-a",
                "jobs": {"j1": {"kind": "bedrock_import", "status": "IN_PROGRESS"}},
            }],
            "LastEvaluatedKey": {"task_id": "thr-a"},
        },
        {
            "Items": [{
                "task_id": "thr-b",
                "jobs": {"j2": {"kind": "bedrock_import", "status": "IN_PROGRESS"}},
            }],
        },
    ]
    p.boto3.resource = lambda *_a, **_kw: MagicMock(Table=lambda _n: fake_table)  # type: ignore[attr-defined]
    matches = p._scan_in_flight_imports()
    ids = sorted(m["thread_id"] for m in matches)
    assert ids == ["thr-a", "thr-b"]
    assert fake_table.scan.call_count == 2


# ---------- _check_one ----------

def _make_check_env(p, *, bedrock_status: str, failure: str = "") -> dict:
    """Wire up minimal stubs so _check_one can run end-to-end in-memory."""
    bedrock_client = MagicMock()
    desc = {"status": bedrock_status}
    if failure:
        desc["failureMessage"] = failure
    bedrock_client.get_model_import_job.return_value = desc

    fake_table = MagicMock()
    update_calls: list = []
    fake_table.update_item = lambda **kw: update_calls.append(kw) or {}

    resume_calls: list = []

    def _client(name, region_name=None):
        if name == "bedrock":
            return bedrock_client
        raise AssertionError(f"unexpected client {name}")

    def _resource(*_a, **_kw):
        return MagicMock(Table=lambda _n: fake_table)

    p.boto3.client = _client          # type: ignore[attr-defined]
    p.boto3.resource = _resource      # type: ignore[attr-defined]
    p._get_agentcore_endpoint = lambda: "https://endpoint.example"  # type: ignore[attr-defined]
    p._invoke_agentcore = lambda endpoint, thread_id, msg, user_id="": resume_calls.append(  # type: ignore[attr-defined]
        {"endpoint": endpoint, "thread_id": thread_id, "msg": msg, "user_id": user_id}
    )
    p._build_resume_message = lambda *a, **kw: "RESUME"  # type: ignore[attr-defined]

    return {"updates": update_calls, "resumes": resume_calls, "bedrock": bedrock_client}


def test_check_one_in_flight_status_does_nothing():
    """InProgress / Submitted / Queued are no-ops (no DDB write, no resume)."""
    p = _import_poller()
    for s in ("InProgress", "Submitted", "Queued"):
        env = _make_check_env(p, bedrock_status=s)
        p._check_one(
            thread_id="t",
            job_id="j",
            entry={"import_job_identifier": "id-x"},
            user_id="u",
        )
        assert env["updates"] == [], f"unexpected DDB write for {s}"
        assert env["resumes"] == [], f"unexpected resume for {s}"


def test_check_one_unknown_status_is_treated_as_in_flight():
    """Defensive: unrecognised statuses must NOT close the row."""
    p = _import_poller()
    env = _make_check_env(p, bedrock_status="Mystery")
    p._check_one(
        thread_id="t",
        job_id="j",
        entry={"import_job_identifier": "id-x"},
        user_id="u",
    )
    assert env["updates"] == []
    assert env["resumes"] == []


def test_check_one_terminal_completed_writes_ddb_and_resumes():
    """Completed must update DDB to COMPLETED and invoke AgentCore resume."""
    p = _import_poller()
    env = _make_check_env(p, bedrock_status="Completed")
    p._check_one(
        thread_id="thr-1",
        job_id="job-1",
        entry={
            "import_job_identifier": "id-x",
            "bedrock_model_name": "myimport",
        },
        user_id="user-1",
    )
    assert len(env["updates"]) == 1
    upd = env["updates"][0]
    assert upd["Key"] == {"task_id": "thr-1"}
    assert upd["ExpressionAttributeNames"]["#jid"] == "job-1"
    # status is uppercased on the way into DDB
    assert upd["ExpressionAttributeValues"][":s"] == "COMPLETED"

    assert len(env["resumes"]) == 1
    assert env["resumes"][0]["thread_id"] == "thr-1"
    assert env["resumes"][0]["user_id"] == "user-1"


def test_check_one_terminal_failed_persists_failure_message():
    """Failed status must capture failureMessage in jobs.<id>.status_message."""
    p = _import_poller()
    env = _make_check_env(p, bedrock_status="Failed", failure="S3 access denied")
    p._check_one(
        thread_id="thr-x",
        job_id="job-x",
        entry={"import_job_identifier": "id-x"},
        user_id="u",
    )
    upd = env["updates"][0]
    assert upd["ExpressionAttributeValues"][":s"] == "FAILED"
    assert upd["ExpressionAttributeValues"][":m"] == "S3 access denied"
    assert "status_message" in upd["UpdateExpression"]


def test_check_one_skips_when_identifier_missing():
    """If neither identifier nor ARN are present, log+skip (no Bedrock call)."""
    p = _import_poller()
    env = _make_check_env(p, bedrock_status="Completed")
    p._check_one(thread_id="t", job_id="j", entry={}, user_id="u")
    env["bedrock"].get_model_import_job.assert_not_called()
    assert env["updates"] == []
    assert env["resumes"] == []


# ---------- handler() ----------

def test_handler_invokes_check_one_per_row():
    """handler must dispatch _check_one for every row returned by the scan."""
    p = _import_poller()
    p._scan_in_flight_imports = lambda: [  # type: ignore[attr-defined]
        {"thread_id": "t1", "job_id": "j1", "entry": {}, "user_id": "u1"},
        {"thread_id": "t2", "job_id": "j2", "entry": {}, "user_id": "u2"},
    ]
    seen: list = []
    p._check_one = lambda **kw: seen.append(kw)  # type: ignore[attr-defined]
    out = p.handler({}, None)
    assert out == {"statusCode": 200, "checked": 2}
    assert [s["thread_id"] for s in seen] == ["t1", "t2"]
