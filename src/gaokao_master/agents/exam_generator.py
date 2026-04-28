"""Exam generation sub-agent for Gaokao-Master."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph
from loguru import logger

from gaokao_master.kb.manager import KnowledgeBaseManager
from gaokao_master.llm import OpenAICompatibleLLM
from gaokao_master.tools import RetrievalHit, fuzzy_retrieve


@dataclass(frozen=True)
class ExamGenerationRequest:
    """Input expected by the exam generator sub-agent."""

    weak_points: list[str]
    subject: str
    topic: str | None = None
    mastered_keywords: list[str] = field(default_factory=list)
    question_count: int = 8
    difficulty: str = "mixed"
    output_dir: str | Path = "outputs/mock_exams"


@dataclass(frozen=True)
class ExamGenerationResult:
    """Generated mock exam and provenance."""

    markdown: str
    output_path: Path
    selected_hits: list[RetrievalHit]


class ExamGraphState(TypedDict):
    """LangGraph state used by ExamGeneratorAgent."""

    request: ExamGenerationRequest
    hits: list[RetrievalHit]
    selected_hits: list[RetrievalHit]
    markdown: str
    output_path: str


class ExamGeneratorAgent:
    """Sub-agent that retrieves and composes personalized mock exams."""

    def __init__(
        self,
        kb_manager: KnowledgeBaseManager | None = None,
        llm_client: OpenAICompatibleLLM | None = None,
    ) -> None:
        self.kb_manager = kb_manager or KnowledgeBaseManager()
        self.llm_client = llm_client
        self.graph = self._build_graph()

    def generate(self, request: ExamGenerationRequest) -> ExamGenerationResult:
        """Generate a custom mock exam from weak-point descriptions."""

        initial_state: ExamGraphState = {
            "request": request,
            "hits": [],
            "selected_hits": [],
            "markdown": "",
            "output_path": "",
        }
        final_state = self.graph.invoke(initial_state)
        return ExamGenerationResult(
            markdown=final_state["markdown"],
            output_path=Path(final_state["output_path"]),
            selected_hits=final_state["selected_hits"],
        )

    def _build_graph(self):
        graph = StateGraph(ExamGraphState)
        graph.add_node("retrieve_questions", self._retrieve_questions)
        graph.add_node("filter_mastered", self._filter_mastered)
        graph.add_node("compose_paper", self._compose_paper)
        graph.add_node("save_paper", self._save_paper)

        graph.set_entry_point("retrieve_questions")
        graph.add_edge("retrieve_questions", "filter_mastered")
        graph.add_edge("filter_mastered", "compose_paper")
        graph.add_edge("compose_paper", "save_paper")
        graph.add_edge("save_paper", END)
        return graph.compile()

    def _retrieve_questions(self, state: ExamGraphState) -> ExamGraphState:
        request = state["request"]
        combined_hits: list[RetrievalHit] = []

        for weak_point in request.weak_points:
            query = (
                f"{request.subject} {request.topic or ''} {weak_point} "
                "高考 试题 题目 选择题 填空题 解答题"
            )
            hits = fuzzy_retrieve(
                query,
                kb_manager=self.kb_manager,
                subject=request.subject or None,
                topic=request.topic,
                top_k=max(request.question_count * 3, 12),
            )
            combined_hits.extend(hits)

        deduped: dict[str, RetrievalHit] = {}
        for hit in combined_hits:
            deduped[hit.chunk_id] = hit

        state["hits"] = sorted(
            deduped.values(),
            key=lambda item: item.score,
            reverse=True,
        )
        return state

    def _filter_mastered(self, state: ExamGraphState) -> ExamGraphState:
        request = state["request"]
        mastered_keywords = [
            keyword.strip().lower()
            for keyword in request.mastered_keywords
            if keyword.strip()
        ]
        selected: list[RetrievalHit] = []

        for hit in state["hits"]:
            question_text = _clean_question_text(hit.text)
            if not question_text or _looks_like_answer_only_block(hit.text):
                continue

            haystack = f"{hit.text} {hit.metadata}".lower()
            if any(keyword in haystack for keyword in mastered_keywords):
                continue
            selected.append(hit)
            if len(selected) >= request.question_count:
                break

        state["selected_hits"] = selected
        return state

    def _compose_paper(self, state: ExamGraphState) -> ExamGraphState:
        request = state["request"]
        title = f"{request.subject}个性化模拟卷"
        if request.topic:
            title = f"{request.subject}-{request.topic}个性化模拟卷"

        lines = [
            f"# {title}",
            "",
            f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- 薄弱点：{'、'.join(request.weak_points) or '未指定'}",
            f"- 难度策略：{request.difficulty}",
            f"- 题目数量：{len(state['selected_hits'])}",
            "",
            "## 答题说明",
            "",
            "请先独立完成全部题目，再对照原资料中的答案与解析复盘。每道题下方保留来源，方便回到知识库继续整理错因。",
            "",
            "## 试题",
            "",
        ]

        if not state["selected_hits"]:
            lines.extend(
                [
                    "> 当前知识库中没有检索到足够相关题目。请先导入对应科目和知识点的真题、模拟题或笔记。",
                    "",
                ]
            )
        else:
            for index, hit in enumerate(state["selected_hits"], start=1):
                question_text = _clean_question_text(hit.text)
                lines.extend(
                    [
                        f"### 第 {index} 题",
                        "",
                        question_text,
                        "",
                        f"> 来源：`{hit.source}`；匹配分：{hit.score:.3f}",
                        "",
                    ]
                )

        lines.extend(
            [
                "## 复盘记录",
                "",
                "| 题号 | 正误 | 错因 | 订正要点 | 是否掌握 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for index in range(1, max(len(state["selected_hits"]), 1) + 1):
            lines.append(f"| {index} |  |  |  |  |")

        draft_markdown = "\n".join(lines).strip() + "\n"
        state["markdown"] = self._enhance_with_llm(state, draft_markdown)
        return state

    def _save_paper(self, state: ExamGraphState) -> ExamGraphState:
        request = state["request"]
        output_dir = Path(request.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        topic_part = f"_{request.topic}" if request.topic else ""
        filename = _safe_filename(
            f"{request.subject}{topic_part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_path = output_dir / f"{filename}_{uuid4().hex[:6]}.md"
        output_path.write_text(state["markdown"], encoding="utf-8")
        state["output_path"] = str(output_path)
        return state

    def _enhance_with_llm(self, state: ExamGraphState, draft_markdown: str) -> str:
        """Use an external model to improve paper wording when configured."""

        if not self.llm_client or not self.llm_client.is_configured:
            return draft_markdown

        request = state["request"]
        source_digest = "\n\n".join(
            [
                f"资料 {index}\n来源：{hit.source}\n内容：{_clean_question_text(hit.text)}"
                for index, hit in enumerate(state["selected_hits"], start=1)
            ]
        )
        system_prompt = (
            "你是 Gaokao-Master 的 Exam_Generator_Agent，负责为中国高考学生生成"
            "个性化练习卷。必须忠实使用给定资料，不要编造题目事实、答案或出处。"
            "输出只能是 Markdown。"
        )
        user_prompt = f"""
请基于下列信息优化一份个性化模拟卷。

科目：{request.subject}
专题：{request.topic or "未限定"}
薄弱点：{"、".join(request.weak_points)}
已掌握关键词：{"、".join(request.mastered_keywords) or "无"}
难度策略：{request.difficulty}

要求：
1. 保留题目主体和来源信息。
2. 调整 Markdown 结构，使试卷清晰、适合打印和复盘。
3. 不要凭空新增题目、答案或解析。
4. 若资料不足，请明确提示需要继续导入资料。

当前规则版草稿：
{draft_markdown}

检索资料：
{source_digest}
"""
        try:
            improved = self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=3_500,
            )
        except Exception as exc:
            logger.warning("LLM exam enhancement failed, using draft: {}", exc)
            return draft_markdown

        return improved.strip() + "\n" if improved.strip() else draft_markdown


def _clean_question_text(text: str) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
    text = _drop_answer_only_prefix(text)
    text = _first_valid_question_block(text)
    text = _strip_answer_and_analysis(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _drop_answer_only_prefix(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    skipping = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d{1,3}[.．、]\s*【?\s*(?:答案|解析)", stripped):
            skipping = True
            continue
        if skipping and re.match(r"^\d{1,3}[.．、]\s*【?\s*(?:答案|解析)", stripped):
            continue
        if skipping and _looks_like_question_start(stripped):
            skipping = False
        if not skipping:
            kept.append(line)

    return "\n".join(kept).strip()


def _first_valid_question_block(text: str) -> str:
    candidates = re.split(
        r"(?m)(?=^\s*(?:\d{1,3}[.．、]|第\s*[\d一二三四五六七八九十]+\s*题)\s*)",
        text,
    )
    valid_blocks = [
        candidate.strip()
        for candidate in candidates
        if _looks_like_question_block(candidate)
    ]
    if valid_blocks:
        return valid_blocks[0]
    return text if _looks_like_question_block(text) else ""


def _strip_answer_and_analysis(text: str) -> str:
    answer_pattern = re.compile(
        r"(?m)(?:^|\n)\s*(?:"
        r"【\s*(?:答案|解析|参考答案|解答|详解)\s*】|"
        r"(?:答案|解析|参考答案|解答|详解)\s*[:：]|"
        r"证明如下\s*[:：]"
        r")"
    )
    match = answer_pattern.search(text)
    if match:
        text = text[: match.start()]

    trailing_patterns = [
        r"(?m)^\s*来源\s*[:：].*$",
        r"(?m)^\s*>?\s*匹配分\s*[:：].*$",
    ]
    for pattern in trailing_patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


def _looks_like_answer_only_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    first_lines = "\n".join(stripped.splitlines()[:8])
    numbered_answer_lines = re.findall(
        r"(?m)^\s*\d{1,3}[.．、]\s*【?\s*(?:答案|解析)",
        first_lines,
    )
    if numbered_answer_lines:
        return True

    answer_markers = len(re.findall(r"(?:答案|解析|参考答案|详解)", stripped))
    question_markers = len(
        re.findall(r"(?:已知|设|若|求|证明|如图|下列|选择|填空|解答)", stripped)
    )
    starts_with_answer = re.match(
        r"^\s*(?:【?\s*(?:答案|解析|参考答案|详解)|\d{1,3}[.．、]\s*【?\s*(?:答案|解析))",
        stripped,
    )
    return bool(starts_with_answer and answer_markers >= question_markers)


def _looks_like_question_block(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    if _looks_like_answer_only_block(stripped):
        return False
    has_numbered_question = bool(
        re.search(r"(?m)^\s*\d{1,3}[.．、]\s*(?!【?\s*(?:答案|解析))", stripped)
    )
    has_options = bool(re.search(r"(?m)^\s*[A-D][.．、]", stripped))
    has_choice_blank = bool(re.search(r"[(（]\s*[　\s]*[)）]", stripped))
    has_task_cue = bool(
        re.search(r"(?:已知|设|若|求|证明|如图|下列|填空|解答|计算|问|是否)", stripped)
    )
    return bool(
        (has_numbered_question and has_task_cue)
        or has_options
        or has_choice_blank
    )


def _looks_like_question_start(line: str) -> bool:
    return bool(
        re.match(r"^\d{1,3}[.．、]\s*(?!【?\s*(?:答案|解析))", line)
        and re.search(r"(?:已知|设|若|求|证明|如图|下列|选择|填空|解答|计算)", line)
    )


def _safe_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    return value[:120] or f"exam_{uuid4().hex[:8]}"
