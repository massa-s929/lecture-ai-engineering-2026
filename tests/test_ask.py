from types import SimpleNamespace

from app import ask


def test_chat_reuses_session_and_shows_token_usage(monkeypatch, capsys):
    sent_messages = []
    session = SimpleNamespace(
        send_message=lambda question: (
            sent_messages.append(question)
            or SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=7,
                ),
                text=f"回答: {question}",
            )
        )
    )
    client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: session)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    questions = iter(["LoRAとは？", "それは何の略？", "終了"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask.chat()

    output = capsys.readouterr().out
    assert sent_messages == ["LoRAとは？", "それは何の略？"]
    assert "[入力トークン] 12" in output
    assert "[出力トークン] 7" in output
    assert "回答: それは何の略？" in output