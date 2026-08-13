"""Fixtures for functional tests that submit real SageMaker jobs.

These tests require a live AWS account with the CDK stack deployed. They load
the skill Lambda handler modules directly (the way the Gateway would) and
invoke training + processing jobs end-to-end. Cycle length is deliberately
tiny (max_steps=1, max_rows=2) to keep each run under ~25 min wall time.

When required infra env vars are missing the whole functional package is
skipped, so local development and CI without AWS credentials remain green.
"""
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Env vars the handlers read at import / call time. If any is missing we
# can't meaningfully exercise the real control-plane path.
REQUIRED_ENV = (
    "AWS_REGION",
    "PROJECT_NAME",
    "SESSION_BUCKET",
    "SAGEMAKER_EXECUTION_ROLE_ARN",
    "EVAL_IMAGE_URI",
    "JOBS_TABLE",
)


@pytest.fixture(scope="session", autouse=True)
def _guard_infra() -> None:
    """Skip the whole functional suite when deploy outputs aren't exported."""
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        pytest.skip(
            f"Functional tests require live AWS infra; missing env: {missing}. "
            "Export the CDK deploy outputs (SESSION_BUCKET, SAGEMAKER_EXECUTION_ROLE_ARN, "
            "EVAL_IMAGE_URI, MLFLOW_TRACKING_URI, JOBS_TABLE, PROJECT_NAME, AWS_REGION) "
            "before running.",
            allow_module_level=True,
        )


@pytest.fixture(scope="session")
def aws_region() -> str:
    return os.environ["AWS_REGION"]


@pytest.fixture(scope="session")
def project_name() -> str:
    return os.environ["PROJECT_NAME"]


@pytest.fixture(scope="session")
def jobs_table_name() -> str:
    return os.environ["JOBS_TABLE"]


@pytest.fixture(scope="session")
def sm_client(aws_region: str):
    return boto3.client("sagemaker", region_name=aws_region)


@pytest.fixture(scope="session")
def ddb_table(aws_region: str, jobs_table_name: str):
    return boto3.resource("dynamodb", region_name=aws_region).Table(jobs_table_name)


@pytest.fixture
def skill_handler():
    """Import `lambda/skills/<skill>/handler.py` fresh per test.

    All skill handlers expose the module name `handler`, so the cached entry
    must be dropped between imports to avoid cross-contamination. Mirrors
    tests/test_lambda_handler_dispatch.py:_import_handler.
    """
    def _load(skill: str):
        skill_dir = REPO_ROOT / "lambda" / "skills" / skill
        for p in list(sys.path):
            if "/lambda/skills/" in p:
                sys.path.remove(p)
        sys.path.insert(0, str(skill_dir))
        sys.modules.pop("handler", None)
        return importlib.import_module("handler")

    return _load


@pytest.fixture
def gateway_ctx():
    """Build a minimal Lambda context mimicking the AgentCore Gateway payload."""
    def _ctx(tool_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            client_context=SimpleNamespace(custom={"bedrockAgentCoreToolName": tool_name})
        )

    return _ctx
