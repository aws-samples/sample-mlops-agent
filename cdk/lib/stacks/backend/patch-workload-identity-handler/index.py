"""Custom resource handler: registers OAuth callback URL with AgentCore workload identity.

Called by CloudFormation after the AgentCore Runtime is created. Patches the workload
identity to associate the HuggingFace OAuth provider callback URL.

Environment: Python 3.12, boto3 bundled with Lambda runtime.
"""
import json
import boto3


def handler(event: dict, context) -> dict:
    """CloudFormation custom resource handler.

    Args:
        event: CloudFormation custom resource event with ResourceProperties:
            - RuntimeId: the Runtime ID (workload identity name is resolved from it)
            - HFOAuthCallbackUrl: the callback URL to register
            - Region: AWS region
        context: Lambda context (unused).

    Returns:
        dict with PhysicalResourceId for CloudFormation tracking.
    """
    request_type = event.get("RequestType", "")
    props = event.get("ResourceProperties", {})
    runtime_id = props.get("RuntimeId", "")
    callback_url = props.get("HFOAuthCallbackUrl", "")
    region = props.get("Region", "us-east-1")

    physical_id = f"patch-workload-identity-{runtime_id}"

    if request_type == "Delete":
        # Nothing to clean up — workload identity lifecycle is managed by Runtime
        return {"PhysicalResourceId": physical_id}

    # Skip when callback URL isn't configured yet (first deploy before the
    # post-deploy runbook seeds the SSM param).
    if not callback_url or callback_url == "PLACEHOLDER":
        print(
            f"[patch-workload-identity] skipping — callback_url={callback_url!r} "
            "(run post-deploy runbook to configure)"
        )
        return {"PhysicalResourceId": physical_id}

    client = boto3.client("bedrock-agentcore-control", region_name=region)

    # Resolve workload identity name from the Runtime itself — the service-linked
    # identity is auto-created by AgentCore, so its ARN is only knowable post-create.
    describe = client.get_agent_runtime(agentRuntimeId=runtime_id)
    wi_arn = describe.get("workloadIdentityDetails", {}).get("workloadIdentityArn", "")
    workload_name = wi_arn.split("/")[-1] if wi_arn else ""
    if not workload_name:
        raise RuntimeError(
            f"Runtime {runtime_id!r} has no workloadIdentityDetails.workloadIdentityArn — "
            "cannot register OAuth callback URL."
        )

    try:
        # Confirm workload identity exists before patching (API param is "name", not "workloadIdentityName")
        resp = client.get_workload_identity(name=workload_name)
        print(f"[patch-workload-identity] found: {json.dumps(resp, default=str)[:500]}")
    except client.exceptions.ResourceNotFoundException:
        raise RuntimeError(
            f"Workload identity '{workload_name}' not found. "
            "Ensure Runtime is fully created before this custom resource runs."
        )

    # Associate OAuth callback URL with the workload identity
    client.update_workload_identity(
        name=workload_name,
        oauthCallbackUrls=[callback_url],
    )
    print(f"[patch-workload-identity] registered callback URL for {workload_name}")

    return {"PhysicalResourceId": physical_id, "Data": {"WorkloadIdentityName": workload_name}}
