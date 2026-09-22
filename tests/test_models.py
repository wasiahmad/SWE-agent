from __future__ import annotations

from pydantic import SecretStr

from sweagent.agent.models import GenericAPIModelConfig, get_model
from sweagent.tools.parsing import Identity
from sweagent.tools.tools import ToolConfig
from sweagent.types import History


def test_litellm_mock():
    model = get_model(
        GenericAPIModelConfig(
            name="gpt-4o",
            completion_kwargs={"mock_response": "Hello, world!"},
            api_key=SecretStr("dummy_key"),
            top_p=None,
        ),
        ToolConfig(
            parse_function=Identity(),
        ),
    )
    assert model.query(History([{"role": "user", "content": "Hello, world!"}])) == {"message": "Hello, world!"}  # type: ignore


def test_litellm_history_preserves_reasoning_for_provider_forwarding():
    model = get_model(
        GenericAPIModelConfig(
            name="gpt-4o",
            api_key=SecretStr("dummy_key"),
            top_p=None,
        ),
        ToolConfig(parse_function=Identity()),
    )
    history = History(
        [
            {
                "role": "assistant",
                "content": "I found the relevant file.",
                "message_type": "action",
                "reasoning_content": "First inspect the repository structure.",
            },
            {
                "role": "user",
                "content": "Continue.",
                "message_type": "observation",
            },
        ]
    )

    messages = model._history_to_messages(history)  # type: ignore[attr-defined]

    assert messages[0]["reasoning_content"] == "First inspect the repository structure."
    assert messages[0]["provider_specific_fields"] == {
        "reasoning": "First inspect the repository structure."
    }
    assert "reasoning_content" not in messages[1]
    assert "provider_specific_fields" not in messages[1]
