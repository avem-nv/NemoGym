# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from pydantic import ConfigDict


def detect_repeating_pattern(text: str, min_repeats: int = 5, max_pattern_words: int = 10) -> bool:
    """Detect if text ends with a repeating pattern (model degeneration)."""
    if not text or len(text.strip()) == 0:
        return False

    text_end = text[-2000:].strip()
    words = text_end.split()

    if len(words) < min_repeats:
        return False

    last_words = words[-min_repeats:]
    if len(set(last_words)) == 1:
        return True

    for pattern_len in range(1, min(max_pattern_words + 1, len(words) // min_repeats)):
        pattern = words[-pattern_len:]
        is_repeating = True
        for i in range(min_repeats):
            start_idx = -(pattern_len * (i + 1))
            end_idx = -(pattern_len * i) if i > 0 else None
            segment = words[start_idx:end_idx] if end_idx else words[start_idx:]
            if segment != pattern:
                is_repeating = False
                break
        if is_repeating:
            return True

    last_chars = text_end[-500:]
    for pattern_len in range(1, 50):
        pattern = last_chars[-pattern_len:]
        repeat_count = 0
        pos = len(last_chars) - pattern_len
        while pos >= 0:
            if last_chars[pos : pos + pattern_len] == pattern:
                repeat_count += 1
                pos -= pattern_len
            else:
                break
        if repeat_count >= min_repeats:
            return True

    return False


def detect_long_strings(text: str, min_length: int = 60, min_occurrences: int = 1) -> bool:
    """Detect abnormally long tokens (words concatenated without spaces)."""
    if not text or len(text.strip()) == 0:
        return False

    clean_text = re.sub(r"<[^>]+>", " ", text)
    tokens = clean_text.split()
    long_token_count = sum(1 for token in tokens if len(token) >= min_length)
    return long_token_count >= min_occurrences


class NemotronParseResourcesServerConfig(BaseResourcesServerConfig):
    pass


class NemotronParseVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="allow")


class NemotronParseVerifyResponse(BaseVerifyResponse):
    """Extends the base verify response with the raw parsed content from nemotron-parse."""

    parsed_content: Optional[str] = None
    model_config = ConfigDict(extra="allow")


class NemotronParseResourcesServer(SimpleResourcesServer):
    config: NemotronParseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Register the tool endpoint that nemotron-parse calls via function_call output.
        # simple_agent will POST to /markdown_bbox with the parsed bounding box JSON.
        app.post("/markdown_bbox")(self.markdown_bbox)
        return app

    async def markdown_bbox(self, request: Request) -> str:
        """Passthrough endpoint — accepts any nemotron-parse bounding box output and acknowledges it."""
        return ""

    async def verify(self, body: NemotronParseVerifyRequest) -> NemotronParseVerifyResponse:
        """
        Extract the markdown_bbox function_call arguments from the model response.
        nemotron-parse returns a tool_call (not text), so body.response.output contains
        a function_call item whose arguments hold the bounding box JSON.

        Reward is 1.0 only if the parsed content passes both quality checks:
        - no repeating pattern (model degeneration)
        - no abnormally long concatenated tokens (OCR-like failure)
        """
        parsed_content = None
        for item in body.response.output:
            if hasattr(item, "type") and item.type == "function_call":
                parsed_content = item.arguments
                break

        if parsed_content and (detect_repeating_pattern(parsed_content) or detect_long_strings(parsed_content)):
            reward = 0.0
        else:
            reward = 1.0

        return NemotronParseVerifyResponse(
            **body.model_dump(),
            parsed_content=parsed_content,
            reward=reward,
        )


if __name__ == "__main__":
    NemotronParseResourcesServer.run_webserver()
