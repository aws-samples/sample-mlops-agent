"""Gateway interceptor Lambda — injects user context before Cedar evaluation.

This Lambda runs as a Gateway interceptor on every tool call. It reads the
_user_id argument that the agent always passes and injects _injected_user_id
into the transformed request body. Cedar policies read this value to evaluate
per-user authorization rules.

AgentCore Gateway interceptor contract:
  - Input:  event["mcp"]["gatewayRequest"]["body"]  (original MCP request)
  - Output: event["mcp"]["transformedGatewayRequest"]["body"] (patched request)
"""
from typing import Any


def handler(event: dict, context: Any) -> dict:
    """Extract _user_id from tool arguments and inject into Cedar context.

    Reads `_user_id` from the tool call arguments (injected by the agent)
    and places it at `params._injected_user_id` for Cedar policy evaluation.

    Args:
        event: AgentCore Gateway interceptor event.
        context: Lambda context (unused).

    Returns:
        dict: Interceptor output with transformedGatewayRequest containing
              _injected_user_id in params.
    """
    body: dict = event.get("mcp", {}).get("gatewayRequest", {}).get("body", {})
    params: dict = body.get("params", {})
    arguments: dict = params.get("arguments", {})
    user_id: str = arguments.get("_user_id", "")

    transformed_body = {
        **body,
        "params": {
            **params,
            "_injected_user_id": user_id,
        },
    }

    return {
        "interceptorOutputVersion": "1.0",
        "mcp": {
            "transformedGatewayRequest": {
                "body": transformed_body,
            }
        },
    }
