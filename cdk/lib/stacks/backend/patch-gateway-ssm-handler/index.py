"""Custom Resource Lambda — overwrites Gateway SSM PLACEHOLDER params after CfnGateway deploy."""
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event: dict, context) -> dict:
    """Write real Gateway values to the 3 SSM params pre-created as PLACEHOLDER.

    Args:
        event: CloudFormation Custom Resource event. ResourceProperties must contain:
            GatewayMcpUrl, CognitoTokenEndpoint, MachineClientId,
            GatewayMcpUrlParam, CognitoTokenEndpointParam, MachineClientIdParam.
        context: Lambda context (unused).

    Returns:
        dict: PhysicalResourceId for CloudFormation.
    """
    request_type = event["RequestType"]
    props = event["ResourceProperties"]
    ssm = boto3.client("ssm")

    if request_type in ("Create", "Update"):
        params = {
            props["GatewayMcpUrlParam"]:          props["GatewayMcpUrl"],
            props["CognitoTokenEndpointParam"]:    props["CognitoTokenEndpoint"],
            props["MachineClientIdParam"]:         props["MachineClientId"],
        }
        # Verify all target params exist before writing — they are pre-created by AgentCoreStack.
        # A missing param means AgentCoreStack didn't deploy correctly; fail loudly rather than
        # silently leaving the Gateway in a partially-configured state.
        ssm_exceptions = ssm.exceptions
        for name in params:
            try:
                ssm.get_parameter(Name=name)
            except ssm_exceptions.ParameterNotFound:
                raise RuntimeError(
                    f"SSM param {name!r} not found — AgentCoreStack may not have deployed. "
                    "Deploy AgentCoreStack before GatewayStack."
                )
        for name, value in params.items():
            ssm.put_parameter(Name=name, Value=value, Type="String", Overwrite=True)
            logger.info("Wrote SSM param %s", name)

    elif request_type == "Delete":
        # Revert to PLACEHOLDER to signal Gateway is deconfigured
        for name in (
            props["GatewayMcpUrlParam"],
            props["CognitoTokenEndpointParam"],
            props["MachineClientIdParam"],
        ):
            ssm.put_parameter(Name=name, Value="PLACEHOLDER", Type="String", Overwrite=True)
            logger.info("Reset SSM param %s to PLACEHOLDER", name)

    return {"PhysicalResourceId": "patch-gateway-ssm"}
