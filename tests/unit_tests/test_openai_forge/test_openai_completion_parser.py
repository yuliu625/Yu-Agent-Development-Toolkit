"""
Tests for completion parser.
"""

from __future__ import annotations
import pytest
from loguru import logger

from openai_forge.openai_completion_parser import (
    OpenAICompletionParser,
)

from openai.types import Completion

from typing import TYPE_CHECKING, Mapping
# if TYPE_CHECKING:


class TestOpenAICompletionParser:
    def test_extract_content(
        self,
        completion: Completion | Mapping,
    ) -> None:
        content = OpenAICompletionParser.extract_content(
            completion=completion,
        )
        logger.info(f"Content: \n{content}")

