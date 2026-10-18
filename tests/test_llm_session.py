"""Session memory must be isolated, optional, and updated only after success."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.web.state import web_state
from src.web.tabs.tab_llm import handle_llm_session, reset_llm_session
from src.llm_brain import LocalLLMBrain


@pytest.fixture
def model(monkeypatch):
    model = Mock()
    model.chat.return_value = SimpleNamespace(
        reply="Ответ", engine="test", latency_ms=1, speed_tps=0
    )
    monkeypatch.setattr(web_state, "llm_model", model)
    return model


def test_session_remembers_turns_without_mutating_previous_state(model):
    first = handle_llm_session("", "", "Меня зовут Анна", True, [])
    prior = first[2]
    second = handle_llm_session("Кратко", "", "Как меня зовут?", True, prior)
    assert len(prior) == 2
    assert len(second[2]) == 4
    assert model.chat.call_args.args[0] == [
        {"role": "system", "content": "Кратко"},
        *prior,
        {"role": "user", "content": "Как меня зовут?"},
    ]


def test_disabled_session_does_not_send_or_store_history(model):
    prior = [{"role": "user", "content": "Старый запрос"}]
    result = handle_llm_session("", "Контекст", "Промпт", False, prior)
    assert model.chat.call_args.args[0] == [{"role": "user", "content": "Контекст\nПромпт"}]
    assert result[2:] == ([], [])


def test_error_does_not_enter_history(model):
    model.chat.side_effect = RuntimeError("недоступна")
    prior = [{"role": "user", "content": "Первый запрос"}]
    result = handle_llm_session("", "", "Новый запрос", True, prior)
    assert "недоступна" in result[0]
    assert result[2] == prior


def test_reset_starts_empty_session():
    result = reset_llm_session(True)
    assert result[0] == []
    assert result[1]["value"] == []
    assert result[1]["visible"] is True


@patch("src.llm_brain.generate")
def test_chat_sends_exact_conversation_without_added_prompts(generate):
    with patch.object(LocalLLMBrain, "_check_health", return_value=True):
        brain = LocalLLMBrain()
    messages = [{"role": "user", "content": "Привет"}]
    generate.return_value = {"message": {"content": "Здравствуйте"}}
    result = brain.chat(messages)
    assert generate.call_args.args[1]["messages"] == messages
    assert "prompt" not in generate.call_args.args[1]
    assert result.reply == "Здравствуйте"
