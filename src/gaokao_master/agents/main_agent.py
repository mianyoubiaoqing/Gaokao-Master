"""Main LangGraph agent orchestration for Gaokao-Master."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from gaokao_master.agents.exam_generator import (
    ExamGenerationRequest,
    ExamGenerationResult,
    ExamGeneratorAgent,
)
from gaokao_master.kb.manager import KnowledgeBaseManager
from gaokao_master.llm import OpenAICompatibleLLM
from gaokao_master.tools import RetrievalHit, fuzzy_retrieve


TaskType = Literal["generate_exam", "retrieve"]


@dataclass(frozen=True)
class MainAgentResponse:
    """User-facing response produced by the main agent."""

    task_type: TaskType
    message: str
    retrieval_hits: list[RetrievalHit]
    exam_result: ExamGenerationResult | None = None


class MainGraphState(TypedDict):
    """LangGraph state for the main agent."""

    user_message: str
    task_type: TaskType
    payload: dict[str, Any]
    retrieval_hits: list[RetrievalHit]
    exam_result: ExamGenerationResult | None
    response: MainAgentResponse | None


class MainGaokaoAgent:
    """Central agent that delegates specialized work to sub-agents."""

    def __init__(
        self,
        kb_manager: KnowledgeBaseManager | None = None,
        llm_client: OpenAICompatibleLLM | None = None,
    ) -> None:
        self.kb_manager = kb_manager or KnowledgeBaseManager()
        self.llm_client = llm_client
        self.exam_agent = ExamGeneratorAgent(self.kb_manager, llm_client=llm_client)
        self.graph = self._build_graph()

    def invoke(self, user_message: str, **payload: Any) -> MainAgentResponse:
        """Run the main agent for either retrieval or mock-exam generation."""

        initial_state: MainGraphState = {
            "user_message": user_message,
            "task_type": "retrieve",
            "payload": payload,
            "retrieval_hits": [],
            "exam_result": None,
            "response": None,
        }
        final_state = self.graph.invoke(initial_state)
        response = final_state["response"]
        if response is None:
            raise RuntimeError("Main agent finished without a response.")
        return response

    def _build_graph(self):
        graph = StateGraph(MainGraphState)
        graph.add_node("route_task", self._route_task)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("delegate_exam", self._delegate_exam)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("route_task")
        graph.add_conditional_edges(
            "route_task",
            self._route_edge,
            {
                "retrieve": "retrieve",
                "generate_exam": "delegate_exam",
            },
        )
        graph.add_edge("retrieve", "finalize")
        graph.add_edge("delegate_exam", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    def _route_task(self, state: MainGraphState) -> MainGraphState:
        payload = state["payload"]
        explicit_task = payload.get("task_type")
        if explicit_task in {"retrieve", "generate_exam"}:
            state["task_type"] = explicit_task
            return state

        message = state["user_message"]
        generate_markers = ["组卷", "出题", "模拟卷", "试卷", "练习卷", "薄弱点"]
        state["task_type"] = (
            "generate_exam"
            if any(marker in message for marker in generate_markers)
            else "retrieve"
        )
        return state

    @staticmethod
    def _route_edge(state: MainGraphState) -> TaskType:
        return state["task_type"]

    def _retrieve(self, state: MainGraphState) -> MainGraphState:
        payload = state["payload"]
        state["retrieval_hits"] = fuzzy_retrieve(
            state["user_message"],
            kb_manager=self.kb_manager,
            subject=payload.get("subject"),
            topic=payload.get("topic"),
            top_k=int(payload.get("top_k", 8)),
        )
        return state

    def _delegate_exam(self, state: MainGraphState) -> MainGraphState:
        payload = state["payload"]
        weak_points = payload.get("weak_points") or [state["user_message"]]
        if isinstance(weak_points, str):
            weak_points = [item.strip() for item in weak_points.splitlines() if item.strip()]

        request = ExamGenerationRequest(
            weak_points=weak_points,
            subject=payload.get("subject") or "未分类",
            topic=payload.get("topic"),
            mastered_keywords=payload.get("mastered_keywords") or [],
            question_count=int(payload.get("question_count", 8)),
            difficulty=payload.get("difficulty", "mixed"),
            output_dir=Path(payload.get("output_dir", "outputs/mock_exams")),
        )
        state["exam_result"] = self.exam_agent.generate(request)
        return state

    def _finalize(self, state: MainGraphState) -> MainGraphState:
        if state["task_type"] == "generate_exam":
            result = state["exam_result"]
            if result is None:
                raise RuntimeError("Exam generation was routed but no result was produced.")
            message = f"已生成个性化模拟卷：{result.output_path}"
            state["response"] = MainAgentResponse(
                task_type="generate_exam",
                message=message,
                retrieval_hits=result.selected_hits,
                exam_result=result,
            )
            return state

        hits = state["retrieval_hits"]
        state["response"] = MainAgentResponse(
            task_type="retrieve",
            message=f"找到 {len(hits)} 条相关资料。",
            retrieval_hits=hits,
            exam_result=None,
        )
        return state
