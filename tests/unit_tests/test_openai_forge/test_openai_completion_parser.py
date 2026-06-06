"""
Tests for completion parser.
"""

from __future__ import annotations
import pytest
from loguru import logger

from modules.openai_forge import (
    OpenAICompletionParser,
)

from openai.types import Completion

from typing import Mapping
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

