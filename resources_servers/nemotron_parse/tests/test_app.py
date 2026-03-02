# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)

from resources_servers.nemotron_parse.app import (
    NemotronParseResourcesServer,
    NemotronParseResourcesServerConfig,
    NemotronParseVerifyRequest,
)


@pytest.fixture
def server():
    config = NemotronParseResourcesServerConfig(
        name="nemotron_parse_server",
        host="localhost",
        port=8000,
    )
    return NemotronParseResourcesServer(config=config)


def _make_verify_request(output_items: list) -> NemotronParseVerifyRequest:
    responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "test", "type": "message"}],
    )
    response = NeMoGymResponse(
        id="resp_test",
        created_at=0,
        model="nvidia/nemotron-parse",
        object="response",
        output=output_items,
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )
    return NemotronParseVerifyRequest(
        responses_create_params=responses_create_params,
        response=response,
        verifier_metadata={"image_filename": "test.png"},
    )


@pytest.mark.asyncio
async def test_verify_extracts_function_call(server):
    """verify() should extract the markdown_bbox function_call arguments as parsed_content."""
    bbox_data = [[{"bbox": {"xmin": 0.1, "ymin": 0.1, "xmax": 0.9, "ymax": 0.2}, "text": "Hello", "type": "Title"}]]
    func_call = NeMoGymResponseFunctionToolCall(
        name="markdown_bbox",
        arguments=json.dumps(bbox_data),
        call_id="call_abc123",
        type="function_call",
        id="call_abc123",
        status="completed",
    )
    body = _make_verify_request([func_call])
    result = await server.verify(body)

    assert result.reward == 1.0
    assert result.parsed_content == json.dumps(bbox_data)


@pytest.mark.asyncio
async def test_verify_no_function_call(server):
    """verify() should return reward=1.0 and parsed_content=None when there is no function_call."""
    msg = NeMoGymResponseOutputMessage(
        id="msg_abc",
        role="assistant",
        content=[NeMoGymResponseOutputText(annotations=[], text="fallback text", type="output_text")],
        status="completed",
        type="message",
    )
    body = _make_verify_request([msg])
    result = await server.verify(body)

    assert result.reward == 1.0
    assert result.parsed_content is None


@pytest.mark.asyncio
async def test_markdown_bbox_endpoint(server):
    """The /markdown_bbox endpoint should accept any body and return an empty string."""
    from fastapi.testclient import TestClient

    app = server.setup_webserver()
    client = TestClient(app)

    payload = [[{"bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 0.1}, "text": "Test", "type": "Title"}]]
    response = client.post("/markdown_bbox", json=payload)
    assert response.status_code == 200
