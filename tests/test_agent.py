"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    deferred = result.output.approvals if isinstance(result.output, DeferredToolRequests) else []
    assert len(_parts(result, ToolReturnPart)) + len(deferred) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_learning_log_waits_for_approval_then_saves(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    first_model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=first_model):
        result = agent.run_sync("今日の学びを記録に保存して", model=first_model)

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.approvals
    assert not (tmp_path / "learning_log.md").exists()

    approvals = {
        call.tool_call_id: True
        for call in result.output.approvals
    }
    deferred_results = result.output.build_results(approvals=approvals)
    resume_model = TestModel(call_tools=[])
    with agent.override(model=resume_model):
        completed = agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
            model=resume_model,
        )

    assert isinstance(completed.output, str)
    assert (tmp_path / "learning_log.md").exists()


def test_learning_log_is_not_saved_when_approval_is_denied(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path)
    first_model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=first_model):
        result = agent.run_sync("今日の学びを記録に保存して", model=first_model)

    assert isinstance(result.output, DeferredToolRequests)
    approvals = {call.tool_call_id: False for call in result.output.approvals}
    deferred_results = result.output.build_results(approvals=approvals)
    resume_model = TestModel(call_tools=[])
    with agent.override(model=resume_model):
        agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
            model=resume_model,
        )

    assert not (tmp_path / "learning_log.md").exists()
