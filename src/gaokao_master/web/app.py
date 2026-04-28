"""Local WebUI for Gaokao-Master."""

from __future__ import annotations

import html
import base64
import mimetypes
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gaokao_master.config import AppSettings, load_app_settings, save_app_settings  # noqa: E402
from gaokao_master.kb import DocumentExtractionError, KnowledgeBaseManager  # noqa: E402
from gaokao_master.llm import (  # noqa: E402
    OpenAICompatibleConfig,
    OpenAICompatibleLLM,
    OpenAICompatibleOCR,
)
from gaokao_master.tools import (  # noqa: E402
    fuzzy_retrieve,
    web_resource_scraper,
    workspace_editor,
)


FIGMA_SERVICE_FLOW_URL = (
    "https://www.figma.com/board/kSeEUF0Q6TCnJvF6sWvxW8"
    "?utm_source=other&utm_content=edit_in_figjam&oai_id=&request_id="
    "b88aafc0-1919-4ab3-b1ee-bc27b8b6b2ea"
)


def main() -> None:
    settings = load_app_settings()

    st.set_page_config(
        page_title="Gaokao-Master",
        page_icon="学",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style()

    st.title("Gaokao-Master")
    st.caption("本地高考知识库、错题本、试卷分析与混合检索工作台")

    with st.sidebar:
        st.header("服务配置")
        kb_root = st.text_input("知识库路径", value=settings.kb_root)
        st.link_button("打开 FigJam 服务流程", FIGMA_SERVICE_FLOW_URL)
        st.divider()
        st.subheader("外部大模型")
        use_llm = st.toggle("启用 OpenAI 兼容 API", value=settings.use_llm)
        api_base = st.text_input(
            "Base URL",
            value=settings.openai_base_url,
            disabled=not use_llm,
        )
        model_name = st.text_input(
            "模型名",
            value=settings.openai_model,
            disabled=not use_llm,
        )
        api_key = st.text_input(
            "API Key",
            value=settings.openai_api_key,
            type="password",
            disabled=not use_llm,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.openai_temperature),
            step=0.1,
            disabled=not use_llm,
        )
        st.caption("兼容 OpenAI 官方接口，以及支持 OpenAI Chat Completions 协议的第三方网关。")
        st.divider()
        st.subheader("在线搜索")
        tavily_api_key = st.text_input(
            "Tavily API Key",
            value=settings.tavily_api_key,
            type="password",
            help="可选。填写后在线资源搜索会优先使用 Tavily。",
        )
        st.caption("后续新增的外部服务配置都会放在 WebUI 中填写，.env 仅作为兜底。")
        st.divider()
        st.subheader("OCR 多模态模型")
        use_ocr = st.toggle("自动 OCR 扫描版 PDF", value=settings.use_ocr)
        ocr_base_url = st.text_input(
            "OCR Base URL",
            value=settings.ocr_base_url,
            disabled=not use_ocr,
        )
        ocr_model = st.text_input(
            "OCR 模型名",
            value=settings.ocr_model,
            disabled=not use_ocr,
            help="填写支持图片输入的 OpenAI-compatible 多模态模型。",
        )
        ocr_api_key = st.text_input(
            "OCR API Key",
            value=settings.ocr_api_key,
            type="password",
            disabled=not use_ocr,
        )
        ocr_temperature = st.slider(
            "OCR Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.ocr_temperature),
            step=0.1,
            disabled=not use_ocr,
        )
        ocr_dpi = st.slider(
            "OCR 图片 DPI",
            min_value=120,
            max_value=240,
            value=int(settings.ocr_dpi),
            step=30,
            disabled=not use_ocr,
        )
        ocr_max_pages = st.slider(
            "OCR 最大页数",
            min_value=1,
            max_value=80,
            value=int(settings.ocr_max_pages),
            disabled=not use_ocr,
        )
        ocr_extract_media = st.toggle(
            "OCR 时自动提取题图到媒体库",
            value=settings.ocr_extract_media,
            disabled=not use_ocr,
        )
        st.caption("当 PDF 无法提取可复制文本时，会自动渲染页面并调用该模型 OCR。")
        st.divider()
        st.write("知识库结构")
        st.code("Gaokao_KB/{subject}/{topic}", language="text")
        embedding_model_name = st.text_input(
            "Embedding 模型",
            value=settings.embedding_model_name,
            help="默认 local-hash 完全离线、无需下载模型。可填 chromadb-default 或 sentence-transformers 模型名。",
        )

        current_settings = AppSettings(
            kb_root=kb_root,
            embedding_model_name=embedding_model_name,
            use_llm=use_llm,
            openai_base_url=api_base,
            openai_model=model_name,
            openai_api_key=api_key,
            openai_temperature=float(temperature),
            tavily_api_key=tavily_api_key,
            use_ocr=use_ocr,
            ocr_base_url=ocr_base_url,
            ocr_model=ocr_model,
            ocr_api_key=ocr_api_key,
            ocr_temperature=float(ocr_temperature),
            ocr_dpi=int(ocr_dpi),
            ocr_max_pages=int(ocr_max_pages),
            ocr_extract_media=bool(ocr_extract_media),
        )
        settings_path = save_app_settings(current_settings)
        st.caption(f"配置已自动保存：{settings_path}")

        if st.button("重建向量索引", use_container_width=True):
            manager = _get_kb_manager(
                kb_root=kb_root,
                embedding_model_name=embedding_model_name,
                use_ocr=use_ocr,
                ocr_base_url=ocr_base_url,
                ocr_model=ocr_model,
                ocr_api_key=ocr_api_key,
                ocr_temperature=ocr_temperature,
                ocr_dpi=ocr_dpi,
                ocr_max_pages=ocr_max_pages,
                ocr_extract_media=ocr_extract_media,
            )
            with st.spinner("正在从 Markdown 重建索引..."):
                total_chunks = manager.rebuild_index()
            st.success(f"已重建 {total_chunks} 个文本块。")

    manager = _get_kb_manager(
        kb_root=kb_root,
        embedding_model_name=embedding_model_name,
        use_ocr=use_ocr,
        ocr_base_url=ocr_base_url,
        ocr_model=ocr_model,
        ocr_api_key=ocr_api_key,
        ocr_temperature=ocr_temperature,
        ocr_dpi=ocr_dpi,
        ocr_max_pages=ocr_max_pages,
        ocr_extract_media=ocr_extract_media,
    )
    llm_client = _build_llm_client(
        enabled=use_llm,
        api_key=api_key,
        base_url=api_base,
        model=model_name,
        temperature=temperature,
    )
    tabs = st.tabs(
        ["资料导入", "智能检索", "错题本", "试卷分析", "在线资源", "工作区"]
    )
    with tabs[0]:
        _render_ingestion_tab(manager)
    with tabs[1]:
        _render_retrieval_tab(manager)
    with tabs[2]:
        _render_wrong_notebook_tab(manager, llm_client=llm_client)
    with tabs[3]:
        _render_paper_analysis_tab(manager, llm_client=llm_client)
    with tabs[4]:
        _render_web_scraper_tab(manager, tavily_api_key=tavily_api_key)
    with tabs[5]:
        _render_workspace_tab(manager)


def _get_kb_manager(
    *,
    kb_root: str,
    embedding_model_name: str,
    use_ocr: bool,
    ocr_base_url: str,
    ocr_model: str,
    ocr_api_key: str,
    ocr_temperature: float,
    ocr_dpi: int,
    ocr_max_pages: int,
    ocr_extract_media: bool,
) -> KnowledgeBaseManager:
    ocr_client = _build_ocr_client(
        enabled=use_ocr,
        api_key=ocr_api_key,
        base_url=ocr_base_url,
        model=ocr_model,
        temperature=float(ocr_temperature),
    )
    return KnowledgeBaseManager(
        kb_root=kb_root,
        embedding_model_name=embedding_model_name or "local-hash",
        ocr_client=ocr_client,
        ocr_dpi=int(ocr_dpi),
        ocr_max_pages=int(ocr_max_pages),
        ocr_extract_media=bool(ocr_extract_media),
    )


def _build_llm_client(
    *,
    enabled: bool,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
) -> OpenAICompatibleLLM | None:
    if not enabled:
        return None
    env_config = OpenAICompatibleConfig.from_env()
    config = OpenAICompatibleConfig(
        api_key=api_key.strip() or env_config.api_key,
        base_url=base_url.strip() or env_config.base_url,
        model=model.strip() or "gpt-4o-mini",
        temperature=temperature,
    )
    return OpenAICompatibleLLM(config)


def _build_ocr_client(
    *,
    enabled: bool,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
) -> OpenAICompatibleOCR | None:
    if not enabled:
        return None
    config = OpenAICompatibleConfig(
        api_key=api_key.strip() or None,
        base_url=base_url.strip() or None,
        model=model.strip() or "gpt-4o-mini",
        temperature=temperature,
        timeout=120,
    )
    return OpenAICompatibleOCR(config)


def _render_ingestion_tab(manager: KnowledgeBaseManager) -> None:
    st.subheader("导入本地资料")
    st.write("上传高考真题、模拟卷、笔记或答案解析，系统会转成 Markdown 并写入本地向量库。")

    with st.form("ingest_form", clear_on_submit=False):
        left, right = st.columns([1, 1])
        with left:
            uploaded_file = st.file_uploader(
                "选择文件",
                type=["md", "txt", "pdf", "docx"],
            )
            title = st.text_input("资料标题，可留空")
        with right:
            subject = st.text_input("科目", value="语文")
            topic = st.text_input("专题", value="现代文阅读")
            tags = st.text_input("标签，用逗号分隔", value="真题,练习")

        submitted = st.form_submit_button("导入知识库", use_container_width=True)

    if submitted:
        if uploaded_file is None:
            st.warning("请先选择一个文件。")
            return

        upload_dir = (
            manager.paths.raw
            / "uploads"
            / manager._safe_part(subject.strip() or "未分类")
            / manager._safe_part(topic.strip() or "未分类")
        )
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_path = upload_dir / uploaded_file.name
        saved_path.write_bytes(uploaded_file.getbuffer())

        with st.spinner("正在解析、切分并写入向量库..."):
            try:
                ingested = manager.ingest_file(
                    saved_path,
                    subject=subject.strip() or "未分类",
                    topic=topic.strip() or "未分类",
                    title=title.strip() or None,
                    tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
                    keep_raw_copy=False,
                )
            except DocumentExtractionError as exc:
                st.warning(
                    "文件已保存，但没有提取到足够可索引文本。"
                    "这通常是扫描版 PDF，需要先 OCR。"
                )
                if exc.diagnostic_path:
                    st.code(str(exc.diagnostic_path), language="text")
                return
        st.success(f"导入完成：{ingested.chunk_count} 个文本块")
        st.code(str(ingested.markdown_path), language="text")


def _render_retrieval_tab(manager: KnowledgeBaseManager) -> None:
    st.subheader("智能检索")
    st.write("同时使用语义检索和关键词检索，适合查知识点、找题、定位解析。")

    with st.form("retrieve_form"):
        query = st.text_area("检索问题", value="函数单调性 高考 真题 解析", height=100)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            subject = st.text_input("限定科目，可留空", value="")
        with col2:
            topic = st.text_input("限定专题，可留空", value="")
        with col3:
            top_k = st.slider("返回数量", min_value=3, max_value=20, value=8)
        submitted = st.form_submit_button("开始检索", use_container_width=True)

    if submitted:
        with st.spinner("正在检索知识库..."):
            hits = fuzzy_retrieve(
                query,
                kb_manager=manager,
                subject=subject.strip() or None,
                topic=topic.strip() or None,
                top_k=top_k,
            )
        _render_hits(hits)


def _render_wrong_notebook_tab(
    manager: KnowledgeBaseManager,
    *,
    llm_client: OpenAICompatibleLLM | None,
) -> None:
    st.subheader("错题本")
    st.write("把真正做错的题沉淀成可复盘的 Markdown 记录，按科目、专题、状态追踪。")

    add_tab, review_tab = st.tabs(["新增错题", "复盘错题"])

    with add_tab:
        with st.form("wrong_question_form", clear_on_submit=False):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                subject = st.text_input("科目", value="数学")
                topic = st.text_input("专题", value="函数")
            with col2:
                source = st.text_input("来源", value="")
                status = st.selectbox("状态", ["待复盘", "已订正", "已掌握"])
            with col3:
                priority = st.selectbox("优先级", ["高", "中", "低"], index=1)
                tags = st.text_input("标签，用逗号分隔", value="错题")

            question = st.text_area("题目", height=160)
            my_answer = st.text_area("我的答案/错误过程", height=110)
            correct_answer = st.text_area("正确答案/标准解析", height=110)
            mistake_reason = st.text_area("错因", height=90)
            correction = st.text_area("订正要点/下次提醒", height=90)
            use_llm_diagnosis = st.checkbox(
                "用大模型补全错因诊断与复习建议",
                value=bool(llm_client and llm_client.is_configured),
                disabled=not bool(llm_client and llm_client.is_configured),
            )
            submitted = st.form_submit_button("保存到错题本", use_container_width=True)

        if submitted:
            if not question.strip():
                st.warning("请至少填写题目。")
                return

            llm_diagnosis = ""
            if use_llm_diagnosis and llm_client and llm_client.is_configured:
                with st.spinner("正在生成错因诊断..."):
                    llm_diagnosis = _diagnose_wrong_question(
                        llm_client=llm_client,
                        question=question,
                        my_answer=my_answer,
                        correct_answer=correct_answer,
                        mistake_reason=mistake_reason,
                    )

            markdown = _build_wrong_question_markdown(
                subject=subject.strip() or "未分类",
                topic=topic.strip() or "未分类",
                source=source.strip(),
                status=status,
                priority=priority,
                tags=[item.strip() for item in tags.split(",") if item.strip()],
                question=question,
                my_answer=my_answer,
                correct_answer=correct_answer,
                mistake_reason=mistake_reason,
                correction=correction,
                llm_diagnosis=llm_diagnosis,
            )
            saved_path = _save_wrong_question(
                manager,
                subject=subject.strip() or "未分类",
                topic=topic.strip() or "未分类",
                markdown=markdown,
            )
            st.success("错题已保存。")
            st.code(str(saved_path), language="text")

    with review_tab:
        wrong_files = _list_wrong_question_files(manager.paths.root)
        if not wrong_files:
            st.info("错题本还没有记录。")
            return

        filters = st.columns([1, 1, 1])
        with filters[0]:
            subject_filter = st.text_input("按科目筛选", value="")
        with filters[1]:
            status_filter = st.selectbox("按状态筛选", ["全部", "待复盘", "已订正", "已掌握"])
        with filters[2]:
            keyword_filter = st.text_input("关键词", value="")

        filtered_files = _filter_wrong_question_files(
            manager.paths.root,
            wrong_files,
            subject_filter=subject_filter,
            status_filter=status_filter,
            keyword_filter=keyword_filter,
        )
        st.caption(f"共 {len(filtered_files)} 条错题记录")
        if not filtered_files:
            st.warning("没有匹配的错题。")
            return

        selected = st.selectbox("选择错题", filtered_files, key="wrong_question_file")
        selected_path = manager.paths.root / selected
        content = selected_path.read_text(encoding="utf-8")
        st.markdown(content)

        update_cols = st.columns([1, 2])
        with update_cols[0]:
            new_status = st.selectbox("更新状态", ["待复盘", "已订正", "已掌握"], key="wrong_status_update")
        with update_cols[1]:
            review_note = st.text_input("复盘备注", value="")

        if st.button("保存复盘状态", use_container_width=True):
            updated = _update_wrong_question_status(content, new_status, review_note)
            selected_path.write_text(updated, encoding="utf-8")
            st.success("复盘状态已更新。")

        confirm_delete = st.checkbox("确认删除该错题")
        if st.button("删除该错题", use_container_width=True):
            if not confirm_delete:
                st.warning("请先勾选确认删除。")
            else:
                deleted_vectors = manager.delete_vectors_for_markdown(selected_path)
                selected_path.unlink()
                st.warning(f"已删除错题，并删除 {deleted_vectors} 条向量记录。")


def _render_paper_analysis_tab(
    manager: KnowledgeBaseManager,
    *,
    llm_client: OpenAICompatibleLLM | None,
) -> None:
    st.subheader("试卷分析")
    st.write("选择一份试卷，分析题型结构、答案解析完整度、公式图片情况，并可用大模型生成复习建议。")

    files = workspace_editor("list", kb_root=manager.paths.root)["files"]
    if not files:
        st.info("当前知识库还没有 Markdown 试卷。")
        return

    selected_file = st.selectbox("选择试卷", files, key="analysis_md")
    paper_markdown = workspace_editor(
        "read",
        selected_file,
        kb_root=manager.paths.root,
    )["content"]
    metrics = _analyze_paper_structure(paper_markdown)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("题目数", str(metrics["question_count"]))
    col2.metric("选择题特征", str(metrics["choice_like_count"]))
    col3.metric("图片数", str(metrics["image_count"]))
    col4.metric("答案/解析标记", str(metrics["answer_marker_count"]))

    st.write("结构摘要")
    st.markdown(_paper_metrics_to_markdown(metrics))

    focus = st.text_area(
        "分析重点",
        value="请分析这套试卷的知识点分布、难度梯度、易错点、复习优先级，并给出一周内的复盘计划。",
        height=110,
    )
    if st.button("生成试卷分析报告", use_container_width=True):
        with st.spinner("正在分析试卷..."):
            report = _generate_paper_analysis_report(
                llm_client=llm_client,
                paper_markdown=paper_markdown,
                metrics=metrics,
                focus=focus,
            )
        output_path = _save_paper_analysis(
            manager.paths.root,
            paper_relative_path=selected_file,
            report=report,
        )
        st.success("试卷分析已生成。")
        st.markdown(report)
        st.caption(f"报告已保存：{output_path}")


def _render_web_scraper_tab(
    manager: KnowledgeBaseManager,
    *,
    tavily_api_key: str = "",
) -> None:
    st.subheader("在线资源")
    st.write("搜索并下载公开高考模拟卷资源，PDF 和 DOCX 会自动进入知识库处理流程。")
    st.info("如果搜索引擎在当前网络下不可用，可以直接粘贴 PDF/DOCX 下载链接。")

    with st.form("scraper_form"):
        query = st.text_input("搜索关键词", value="2025 高考 数学 模拟卷")
        manual_urls = st.text_area(
            "手动资源链接，可留空；每行一个 PDF/DOCX/DOC 或包含下载链接的网页",
            value="",
            height=96,
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            subject = st.text_input("科目", value="数学", key="scraper_subject")
        with col2:
            topic = st.text_input("专题", value="综合模拟", key="scraper_topic")
        with col3:
            max_results = st.slider("最多下载", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("搜索并导入", use_container_width=True)

    if submitted:
        with st.spinner("正在搜索、下载并导入资源..."):
            try:
                resources = web_resource_scraper(
                    query=query,
                    subject=subject.strip() or "未分类",
                    topic=topic.strip() or "综合模拟",
                    kb_manager=manager,
                    max_results=max_results,
                    tavily_api_key=tavily_api_key.strip() or None,
                    resource_urls=[
                        item.strip()
                        for item in manual_urls.splitlines()
                        if item.strip()
                    ],
                )
            except Exception as exc:
                st.error("在线资源导入失败。请检查链接是否可访问，或改用手动上传资料。")
                st.exception(exc)
                return
        if not resources:
            st.warning(
                "没有找到可下载的 PDF/DOCX 资源。当前网络可能无法访问搜索引擎，"
                "建议粘贴直链，或在“资料导入”页手动上传文件。"
            )
            return
        for resource in resources:
            st.write(f"**{resource.title}**")
            st.caption(resource.url)
            st.code(str(resource.local_path), language="text")
            if resource.ingested_markdown_path:
                st.code(str(resource.ingested_markdown_path), language="text")
            st.caption(f"状态：{resource.status}")


def _render_workspace_tab(manager: KnowledgeBaseManager) -> None:
    st.subheader("Markdown 工作区")
    st.write("像 Obsidian 一样预览 Markdown，也可以管理下载区和 RAG 向量记录。")

    preview_tab, edit_tab, media_tab, raw_tab, rag_tab = st.tabs(
        ["Markdown 预览", "源码编辑", "媒体库", "下载区", "RAG 区"]
    )

    files = workspace_editor("list", kb_root=manager.paths.root)["files"]

    with preview_tab:
        if not files:
            st.info("当前知识库还没有 Markdown 文件。")
        else:
            selected_file = st.selectbox("选择 Markdown", files, index=0, key="preview_md")
            content = workspace_editor(
                "read",
                selected_file,
                kb_root=manager.paths.root,
            )["content"]
            st.caption(selected_file)
            components.html(
                _markdown_to_obsidian_document(
                    content,
                    kb_root=manager.paths.root,
                    markdown_relative_path=selected_file,
                ),
                height=760,
                scrolling=True,
            )

    with edit_tab:
        if not files:
            st.info("当前知识库还没有 Markdown 文件。")
        else:
            selected_file = st.selectbox("选择 Markdown", files, index=0, key="edit_md")
            content = workspace_editor(
                "read",
                selected_file,
                kb_root=manager.paths.root,
            )["content"]
            edited = st.text_area("源码内容", value=content, height=480)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("保存修改", use_container_width=True):
                    workspace_editor(
                        "write",
                        selected_file,
                        content=edited,
                        kb_root=manager.paths.root,
                    )
                    st.success("已保存。")
            with col2:
                delete_vectors = st.checkbox("同时删除该文件的 RAG 向量记录")
                confirm_delete = st.checkbox("确认删除该 Markdown 文件")
                if st.button("删除 Markdown", use_container_width=True):
                    if not confirm_delete:
                        st.warning("请先勾选确认删除。")
                    else:
                        target_path = (manager.paths.root / selected_file).resolve()
                        deleted_vectors = 0
                        if delete_vectors:
                            deleted_vectors = manager.delete_vectors_for_markdown(target_path)
                        workspace_editor(
                            "delete",
                            selected_file,
                            kb_root=manager.paths.root,
                        )
                        st.warning(f"已删除 Markdown；同时删除 {deleted_vectors} 条向量记录。")

    with media_tab:
        _render_media_library_tab(manager.paths.root)

    with raw_tab:
        st.write("新导入的原始文件会按 `_raw/web/科目/专题` 或 `_raw/uploads/科目/专题` 保存，并避免重复复制。")
        confirm_compact_raw = st.checkbox("确认删除 _raw 区内容完全相同的重复文件")
        if st.button("整理 _raw 重复文件", use_container_width=True):
            if not confirm_compact_raw:
                st.warning("请先勾选确认后再整理。")
            else:
                deleted_count, freed_bytes = _deduplicate_raw_files(manager.paths.raw)
                st.success(
                    f"已删除 {deleted_count} 个重复原始文件，释放 {_format_file_size(freed_bytes)}。"
                )

        raw_files = _list_raw_files(manager.paths.root)
        if not raw_files:
            st.info("下载区暂无文件。")
        else:
            selected_raw = st.selectbox("选择下载区文件", raw_files, key="raw_file")
            raw_path = _safe_child_path(manager.paths.raw, selected_raw)
            st.caption(str(raw_path))
            st.write(f"大小：{_format_file_size(raw_path.stat().st_size)}")
            delete_raw_vectors = st.checkbox("同时删除该原始文件对应的 RAG 向量记录")
            confirm_raw = st.checkbox("确认删除该下载区文件")
            if st.button("删除下载区文件", use_container_width=True):
                if not confirm_raw:
                    st.warning("请先勾选确认删除。")
                else:
                    deleted_vectors = 0
                    if delete_raw_vectors:
                        deleted_vectors = manager.delete_vectors_for_source(raw_path)
                    raw_path.unlink()
                    st.warning(f"已删除下载区文件；同时删除 {deleted_vectors} 条向量记录。")

    with rag_tab:
        st.write(f"当前 Collection：`{manager.collection_name}`")
        if files:
            selected_file = st.selectbox("按 Markdown 删除向量", files, key="rag_md")
            target_path = (manager.paths.root / selected_file).resolve()
            if st.button("仅删除该 Markdown 的 RAG 向量记录", use_container_width=True):
                deleted = manager.delete_vectors_for_markdown(target_path)
                st.success(f"已删除 {deleted} 条向量记录。")
        else:
            st.info("没有可用于按文档删除向量的 Markdown 文件。")

        st.divider()
        st.warning("清空当前 RAG Collection 后，需要重新点击左侧“重建向量索引”。")
        confirm_name = st.text_input("输入当前 Collection 名以确认清空")
        if st.button("清空当前 RAG Collection", use_container_width=True):
            if confirm_name != manager.collection_name:
                st.warning("Collection 名不匹配，未执行清空。")
            else:
                cleared_name = manager.clear_vector_collection()
                st.cache_resource.clear()
                st.success(f"已清空 Collection：{cleared_name}。请刷新页面后重建索引。")


def _render_hits(hits) -> None:
    if not hits:
        st.warning("没有找到相关资料。")
        return

    for index, hit in enumerate(hits, start=1):
        with st.container(border=True):
            st.write(f"**结果 {index}** · 分数 `{hit.score:.3f}`")
            st.caption(f"{hit.subject} / {hit.topic} / {hit.source}")
            st.markdown(hit.text)


def _render_media_library_tab(kb_root: Path) -> None:
    media_root = _media_root(kb_root)
    media_root.mkdir(parents=True, exist_ok=True)

    st.write("上传题图、几何图、函数图像等资源。Markdown 中可用 Obsidian 语法引用。")
    uploaded_files = st.file_uploader(
        "上传媒体文件",
        type=["png", "jpg", "jpeg", "webp", "gif", "svg"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        saved_paths: list[Path] = []
        for uploaded_file in uploaded_files:
            target = _unique_media_path(media_root, uploaded_file.name)
            target.write_bytes(uploaded_file.getbuffer())
            saved_paths.append(target)
        st.success(f"已上传 {len(saved_paths)} 个媒体文件。")
        for path in saved_paths:
            relative = path.relative_to(kb_root).as_posix()
            st.code(f"![[{path.name}]]\n![{path.stem}]({relative})", language="markdown")

    media_files = _list_media_files(kb_root)
    if not media_files:
        st.info("媒体库暂无文件。")
        return

    selected_media = st.selectbox("选择媒体", media_files, key="media_file")
    media_path = _safe_child_path(kb_root, selected_media)
    st.caption(str(media_path))
    if _is_image_file(media_path):
        st.image(str(media_path), use_container_width=True)
    st.write(f"大小：{_format_file_size(media_path.stat().st_size)}")

    st.write("引用方式")
    st.code(
        f"![[{media_path.name}]]\n![{media_path.stem}]({Path(selected_media).as_posix()})",
        language="markdown",
    )

    confirm_media = st.checkbox("确认删除该媒体文件")
    if st.button("删除媒体文件", use_container_width=True):
        if not confirm_media:
            st.warning("请先勾选确认删除。")
        else:
            media_path.unlink()
            st.warning("已删除媒体文件。")


def _diagnose_wrong_question(
    *,
    llm_client: OpenAICompatibleLLM,
    question: str,
    my_answer: str,
    correct_answer: str,
    mistake_reason: str,
) -> str:
    system_prompt = (
        "你是高考错题诊断助手。请基于学生的错题记录，输出简洁、可执行的复盘建议。"
        "不要重新出题，不要编造题目条件。"
    )
    user_prompt = "\n\n".join(
        [
            "## 题目",
            question.strip(),
            "## 我的答案/错误过程",
            my_answer.strip() or "未填写",
            "## 正确答案/解析",
            correct_answer.strip() or "未填写",
            "## 我自己写的错因",
            mistake_reason.strip() or "未填写",
            "请输出：核心错因、暴露的知识点漏洞、订正步骤、三条复盘提醒。",
        ]
    )
    return llm_client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1_200,
    )


def _build_wrong_question_markdown(
    *,
    subject: str,
    topic: str,
    source: str,
    status: str,
    priority: str,
    tags: list[str],
    question: str,
    my_answer: str,
    correct_answer: str,
    mistake_reason: str,
    correction: str,
    llm_diagnosis: str,
) -> str:
    created_at = datetime.now().isoformat(timespec="seconds")
    tag_values = ", ".join(f'"{tag}"' for tag in tags)
    return "\n\n".join(
        [
            "---",
            'type: "wrong_question"',
            f'subject: "{subject}"',
            f'topic: "{topic}"',
            f'source: "{source}"',
            f'status: "{status}"',
            f'priority: "{priority}"',
            f"tags: [{tag_values}]",
            f'created_at: "{created_at}"',
            'reviewed_at: ""',
            "---",
            f"# 错题：{subject} / {topic}",
            "## 题目",
            question.strip(),
            "## 我的答案/错误过程",
            my_answer.strip() or "未填写",
            "## 正确答案/标准解析",
            correct_answer.strip() or "未填写",
            "## 错因",
            mistake_reason.strip() or "未填写",
            "## 订正要点",
            correction.strip() or "未填写",
            "## AI 诊断",
            llm_diagnosis.strip() or "未生成",
            "## 复盘记录",
            "- 尚未复盘",
        ]
    ).strip() + "\n"


def _save_wrong_question(
    manager: KnowledgeBaseManager,
    *,
    subject: str,
    topic: str,
    markdown: str,
) -> Path:
    target_dir = (
        manager.paths.root
        / manager._safe_part(subject)
        / "错题本"
        / manager._safe_part(topic)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{manager._safe_filename(topic)}.md"
    target = target_dir / filename
    target.write_text(markdown, encoding="utf-8")

    chunks = manager.chunk_markdown(markdown)
    manager.upsert_markdown(
        markdown_path=target,
        chunks=chunks,
        subject=subject,
        topic=f"错题本/{topic}",
        source_path=target,
        tags=["wrong-question", topic],
    )
    return target


def _list_wrong_question_files(kb_root: Path) -> list[str]:
    files = []
    for path in kb_root.rglob("*.md"):
        try:
            relative = path.relative_to(kb_root)
        except ValueError:
            continue
        if "错题本" in relative.parts and path.is_file():
            files.append(str(relative))
    return sorted(files)


def _filter_wrong_question_files(
    kb_root: Path,
    files: list[str],
    *,
    subject_filter: str,
    status_filter: str,
    keyword_filter: str,
) -> list[str]:
    subject_filter = subject_filter.strip().lower()
    keyword_filter = keyword_filter.strip().lower()
    filtered: list[str] = []

    for relative in files:
        path = kb_root / relative
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata = _parse_frontmatter(text)
        haystack = f"{relative}\n{text}".lower()
        if subject_filter and subject_filter not in haystack:
            continue
        if status_filter != "全部" and metadata.get("status") != status_filter:
            continue
        if keyword_filter and keyword_filter not in haystack:
            continue
        filtered.append(relative)

    return filtered


def _parse_frontmatter(markdown_text: str) -> dict[str, str]:
    if not markdown_text.startswith("---\n"):
        return {}
    parts = markdown_text.split("---\n", 2)
    if len(parts) != 3:
        return {}

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata


def _update_wrong_question_status(
    markdown_text: str,
    new_status: str,
    review_note: str,
) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    text = re.sub(
        r'(?m)^status:\s*".*?"\s*$',
        f'status: "{new_status}"',
        markdown_text,
    )
    text = re.sub(
        r'(?m)^reviewed_at:\s*".*?"\s*$',
        f'reviewed_at: "{now}"',
        text,
    )
    note = review_note.strip() or f"状态更新为：{new_status}"
    if "## 复盘记录" not in text:
        text += "\n\n## 复盘记录\n"
    text = text.rstrip() + f"\n- {now}：{note}\n"
    return text


def _analyze_paper_structure(markdown_text: str) -> dict[str, object]:
    body = _strip_frontmatter(markdown_text)
    question_starts = re.findall(
        r"(?m)^\s*(?:\d{1,3}[.．、]|第\s*[\d一二三四五六七八九十]+\s*题)",
        body,
    )
    sections = re.findall(r"(?m)^#{1,3}\s+(.+)$|^\s*[一二三四五六七八九十]+[、.．]\s*(.+)$", body)
    section_names = [item for group in sections for item in group if item]
    choice_like_count = len(re.findall(r"(?m)^\s*[A-D][.．、]", body))
    image_count = len(re.findall(r"!\[[^\]]*\]\([^)]+\)|!\[\[[^\]]+\]\]", body))
    formula_count = len(re.findall(r"\$[^$]+\$|\\frac|\\sqrt|\\sum|\\vec", body))
    answer_marker_count = len(re.findall(r"答案|参考答案|解析|详解", body))
    keyword_hits = _guess_paper_keywords(body)

    return {
        "char_count": len(body),
        "question_count": len(question_starts),
        "choice_like_count": choice_like_count,
        "image_count": image_count,
        "formula_count": formula_count,
        "answer_marker_count": answer_marker_count,
        "section_names": section_names[:12],
        "keyword_hits": keyword_hits,
    }


def _guess_paper_keywords(text: str) -> list[tuple[str, int]]:
    candidates = [
        "函数", "导数", "数列", "立体几何", "解析几何", "概率", "统计",
        "集合", "复数", "三角函数", "向量", "阅读", "作文", "文言文",
        "完形填空", "语法填空", "历史解释", "唯物史观", "区域地理",
        "自然地理", "政治经济", "哲学", "文化", "法治",
    ]
    hits = [(keyword, text.count(keyword)) for keyword in candidates]
    return [(keyword, count) for keyword, count in hits if count > 0][:12]


def _paper_metrics_to_markdown(metrics: dict[str, object]) -> str:
    sections = metrics.get("section_names") or []
    keywords = metrics.get("keyword_hits") or []
    lines = [
        f"- 字符数：{metrics['char_count']}",
        f"- 公式/LaTeX 特征：{metrics['formula_count']}",
        f"- 图片引用：{metrics['image_count']}",
        f"- 答案解析标记：{metrics['answer_marker_count']}",
    ]
    if sections:
        lines.append("- 识别到的栏目：" + "、".join(str(item) for item in sections))
    if keywords:
        lines.append(
            "- 高频知识点线索："
            + "、".join(f"{keyword}({count})" for keyword, count in keywords)
        )
    return "\n".join(lines)


def _generate_paper_analysis_report(
    *,
    llm_client: OpenAICompatibleLLM | None,
    paper_markdown: str,
    metrics: dict[str, object],
    focus: str,
) -> str:
    static_report = "\n\n".join(
        [
            "# 试卷分析报告",
            "## 结构统计",
            _paper_metrics_to_markdown(metrics),
        ]
    )
    if not llm_client or not llm_client.is_configured:
        return (
            static_report
            + "\n\n## 复习建议\n"
            + "- 当前未配置外部大模型，已生成结构统计。建议结合答案解析手动标注错题与薄弱专题。\n"
        )

    system_prompt = (
        "你是高考复习规划助手。请分析一份试卷的结构、知识点、难度梯度和复盘优先级。"
        "输出 Markdown，不要编造试卷中不存在的题目。"
    )
    user_prompt = "\n\n".join(
        [
            "## 静态统计",
            _paper_metrics_to_markdown(metrics),
            "## 分析重点",
            focus.strip(),
            "## 试卷正文",
            paper_markdown[:60_000],
        ]
    )
    llm_report = llm_client.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=3_000,
    )
    return llm_report.strip() + "\n" if llm_report.strip() else static_report


def _save_paper_analysis(
    kb_root: Path,
    *,
    paper_relative_path: str,
    report: str,
) -> Path:
    target_dir = kb_root / "试卷分析"
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r'[<>:"/\\|?*\s]+', "_", Path(paper_relative_path).stem)
    target = target_dir / f"{timestamp}_{safe_stem}.md"
    target.write_text(
        "\n\n".join(
            [
                "---",
                'type: "paper_analysis"',
                f'paper: "{paper_relative_path}"',
                f'created_at: "{datetime.now().isoformat(timespec="seconds")}"',
                "---",
                report.strip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def _strip_frontmatter(markdown_text: str) -> str:
    if markdown_text.startswith("---\n"):
        parts = markdown_text.split("---\n", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return markdown_text.strip()


def _markdown_to_obsidian_html(
    markdown_text: str,
    *,
    kb_root: Path,
    markdown_relative_path: str,
) -> str:
    body = _strip_frontmatter(markdown_text)
    body = _normalize_ocr_html_artifacts(body)
    body = _normalize_ocr_latex_for_preview(body)
    body = _convert_latex_math_to_html(body)
    body = _convert_obsidian_markup_to_markdown(body)
    body = _embed_local_media_links(
        body,
        kb_root=kb_root,
        markdown_relative_path=markdown_relative_path,
    )
    body = _render_bare_latex_fragments(body)
    try:
        import markdown as markdown_lib

        rendered = markdown_lib.markdown(
            body,
            extensions=["extra", "tables", "fenced_code", "sane_lists", "nl2br"],
            output_format="html5",
        )
    except Exception:
        escaped = (
            body.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        rendered = f"<pre>{escaped}</pre>"

    return f'<div class="obsidian-preview">{rendered}</div>'


def _normalize_ocr_html_artifacts(markdown_text: str) -> str:
    """Fix common OCR/model artifacts that should render as inline HTML."""

    replacements = {
        "&lt;sup&gt;": "<sup>",
        "&lt;/sup&gt;": "</sup>",
        "&lt;sub&gt;": "<sub>",
        "&lt;/sub&gt;": "</sub>",
        "\\<sup\\>": "<sup>",
        "\\</sup\\>": "</sup>",
        "\\<sub\\>": "<sub>",
        "\\</sub\\>": "</sub>",
    }
    text = markdown_text
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _normalize_ocr_latex_for_preview(markdown_text: str) -> str:
    """Turn OCR-emitted bare LaTeX into renderable inline math."""

    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"@@LATEX_PROTECTED_{len(protected) - 1}@@"

    text = markdown_text
    protection_patterns = [
        r"```.*?```",
        r"`[^`\n]+`",
        r"!\[[^\]]*\]\([^)]+\)",
        r"<img\b[^>]*>",
        r"\$\$.*?\$\$",
        r"\\\[.*?\\\]",
        r"\\\(.*?\\\)",
        r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)",
    ]
    for pattern in protection_patterns:
        text = re.sub(pattern, protect, text, flags=re.DOTALL)

    normalized_lines = [
        _wrap_bare_latex_in_line(_fix_latex_shorthand(line))
        for line in text.splitlines()
    ]
    text = "\n".join(normalized_lines)

    for index, value in enumerate(protected):
        text = text.replace(f"@@LATEX_PROTECTED_{index}@@", value)
    return text


def _fix_latex_shorthand(text: str) -> str:
    """Repair common OCR shortcuts before math conversion."""

    text = text.replace("\\left.", "").replace("\\right.", "")
    text = re.sub(r"\\left\s*([()\[\]{}|])", r"\1", text)
    text = re.sub(r"\\right\s*([()\[\]{}|])", r"\1", text)
    text = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"\\sqrt{\1}", text)
    text = re.sub(r"\\(vec|bar|overline)\s*([A-Za-z])", r"\\\1{\2}", text)
    text = re.sub(r"\\frac\s*([A-Za-z0-9])\s*\{", r"\\frac{\1}{", text)
    text = re.sub(
        r"\\frac\s+([A-Za-z0-9+\-]+)\s+([A-Za-z0-9+\-]+)",
        r"\\frac{\1}{\2}",
        text,
    )
    return text


def _wrap_bare_latex_in_line(line: str) -> str:
    if not line.strip():
        return line

    line = _wrap_parenthesized_math(line)
    segments = re.split(r"(\$[^$\n]+\$)", line)
    rendered: list[str] = []
    for segment in segments:
        if segment.startswith("$") and segment.endswith("$"):
            rendered.append(segment)
        else:
            rendered.append(_wrap_plain_script_math(_wrap_command_math(segment)))
    return "".join(rendered).replace("$$", "$ $")


def _wrap_parenthesized_math(line: str) -> str:
    line = re.sub(
        r"(?<![$\\])\((\|[^()\n]+\|\s*=\s*\(\s*\))\)",
        lambda match: f"$({match.group(1)})$",
        line,
    )
    pattern = re.compile(r"(?<![$\\])\(([^()\n]{1,120})\)")

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not _looks_like_math_fragment(inner):
            return match.group(0)
        return f"$({inner})$"

    return pattern.sub(replace, line)


def _wrap_command_math(line: str) -> str:
    commands = (
        "frac|sqrt|sum|vec|bar|overline|log|ln|sin|cos|tan|left|right|cdot|"
        "cap|cup|in|notin|leq|geq|le|ge|neq|ne|perp|parallel|angle|triangle|"
        "pi|lambda|mu|alpha|beta|gamma|theta|omega|Delta"
    )
    pattern = re.compile(
        rf"(?<![$\\])((?:\\(?:{commands})|[A-Za-z]\\(?:cdot|cap|cup))"
        r"[^,，。；;：:、\n]{0,120})"
    )

    def replace(match: re.Match[str]) -> str:
        fragment = match.group(1).rstrip()
        trailing = match.group(1)[len(fragment):]
        fragment = _trim_fragment_at_option_boundary(fragment)
        if not fragment or not _looks_like_math_fragment(fragment):
            return match.group(0)
        rest = match.group(1)[len(fragment):] + trailing
        return f"${fragment}$" + rest

    return pattern.sub(replace, line)


def _wrap_plain_script_math(line: str) -> str:
    pattern = re.compile(
        r"(?<![$A-Za-z0-9])([A-Za-z](?:_\{?[\w+\-]+\}?|\^\{?[\w+\-]+\}?)(?:\s*[<>=+\-]\s*[\w{}+\-.]+)?)"
    )
    return pattern.sub(lambda match: f"${match.group(1)}$", line)


def _looks_like_math_fragment(value: str) -> bool:
    if not value or value.startswith("@@LATEX_PROTECTED_"):
        return False
    if re.search(r"\\[A-Za-z]+", value):
        return True
    if re.search(r"[A-Za-z0-9]\s*[_^]\s*\{?[A-Za-z0-9+\-]+", value):
        return True
    if re.search(r"[A-Za-z0-9{}|]\s*(?:[=<>+\-*/]|\\cdot|\\cap|\\cup)\s*[A-Za-z0-9{}|]", value):
        return True
    return False


def _trim_fragment_at_option_boundary(fragment: str) -> str:
    option_match = re.search(r"\s+[A-D][.．]\s*", fragment)
    if option_match:
        return fragment[: option_match.start()].rstrip()
    return fragment


def _render_bare_latex_fragments(markdown_text: str) -> str:
    """Render common LaTeX commands that OCR emitted without $ delimiters."""

    segments = re.split(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))", markdown_text, flags=re.DOTALL)
    rendered_segments: list[str] = []
    for segment in segments:
        if not segment:
            continue
        if _is_math_delimited(segment):
            rendered_segments.append(segment)
        else:
            rendered_segments.append(_render_bare_latex_segment(segment))
    return "".join(rendered_segments)


def _is_math_delimited(segment: str) -> bool:
    return (
        segment.startswith("$$")
        or segment.startswith("$")
        or segment.startswith("\\[")
        or segment.startswith("\\(")
    )


def _render_bare_latex_segment(segment: str) -> str:
    segment = re.sub(
        r"\\sum(?:_\{([^{}]+)\}|_([A-Za-z0-9+\-=]+))?(?:\^\{([^{}]+)\}|\^([A-Za-z0-9+\-=]+))?",
        lambda match: _render_sum_html(
            lower=match.group(1) or match.group(2) or "",
            upper=match.group(3) or match.group(4) or "",
        ),
        segment,
    )
    segment = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        lambda match: _latex_to_basic_html(match.group(0)),
        segment,
    )
    segment = re.sub(
        r"\\frac\s+([A-Za-z0-9+\-]+)\s+\{([^{}]+)\}",
        lambda match: _render_fraction_html(match.group(1), match.group(2)),
        segment,
    )
    segment = re.sub(
        r"\\frac\s+([A-Za-z0-9+\-]+)\s+([A-Za-z0-9+\-]+)",
        lambda match: _render_fraction_html(match.group(1), match.group(2)),
        segment,
    )
    segment = re.sub(
        r"\\sqrt\{([^{}]+)\}",
        lambda match: _latex_to_basic_html(match.group(0)),
        segment,
    )
    replacements = {
        "\\perp": "⊥",
        "\\parallel": "∥",
        "\\leq": "≤",
        "\\geq": "≥",
        "\\le": "≤",
        "\\ge": "≥",
        "\\neq": "≠",
        "\\ne": "≠",
        "\\angle": "∠",
        "\\triangle": "△",
        "\\circ": "°",
    }
    for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
        segment = segment.replace(source, target)
    if re.search(r"[A-Za-z0-9]\^|[A-Za-z]_", segment):
        segment = _render_scripts(segment)
    return segment


def _render_fraction_html(numerator: str, denominator: str) -> str:
    return (
        '<span class="math-frac">'
        f'<span class="math-num">{_latex_to_basic_html(numerator)}</span>'
        f'<span class="math-den">{_latex_to_basic_html(denominator)}</span>'
        "</span>"
    )


def _render_sum_html(lower: str, upper: str) -> str:
    lower_html = _latex_to_basic_html(lower) if lower else ""
    upper_html = _latex_to_basic_html(upper) if upper else ""
    return (
        '<span class="math-sum">'
        f'<span class="math-sum-upper">{upper_html}</span>'
        '<span class="math-sum-symbol">∑</span>'
        f'<span class="math-sum-lower">{lower_html}</span>'
        '</span>'
    )


def _markdown_to_obsidian_document(
    markdown_text: str,
    *,
    kb_root: Path,
    markdown_relative_path: str,
) -> str:
    """Render Markdown in an isolated document with MathJax, like Obsidian."""

    body_html = _markdown_to_obsidian_html(
        markdown_text,
        kb_root=kb_root,
        markdown_relative_path=markdown_relative_path,
    )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true,
        processEnvironments: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    {_obsidian_preview_css()}
  </style>
</head>
<body>
  <button class="print-preview-button" onclick="window.print()">打印试卷</button>
  {body_html}
</body>
</html>
"""


def _obsidian_preview_css() -> str:
    return """
    body {
      margin: 0;
      background: #ffffff;
      color: #111827;
    }
    .print-preview-button {
      position: fixed;
      right: 18px;
      top: 14px;
      z-index: 20;
      border: 1px solid #1f2937;
      background: #ffffff;
      border-radius: 6px;
      padding: 7px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    .obsidian-preview {
      background: #ffffff;
      color: #111827;
      line-height: 1.9;
      font-size: 18px;
      max-width: 980px;
      margin: 0 auto;
      padding: 2rem 3rem 4rem;
      overflow-x: auto;
      font-family: "Times New Roman", "Noto Serif CJK SC", "SimSun", serif;
    }
    .obsidian-preview p {
      margin: .65rem 0 1rem;
    }
    .obsidian-preview h1,
    .obsidian-preview h2,
    .obsidian-preview h3 {
      border-bottom: 0;
      padding-bottom: 0;
      margin: 1.35rem 0 .8rem;
      color: #111827;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    .obsidian-preview h1 { font-size: 1.65rem; }
    .obsidian-preview h2 { font-size: 1.35rem; }
    .obsidian-preview h3 { font-size: 1.15rem; }
    .obsidian-preview table {
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
    }
    .obsidian-preview img {
      display: block;
      max-width: 100%;
      height: auto;
      margin: 1rem auto;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
    }
    .obsidian-preview th,
    .obsidian-preview td {
      border: 1px solid #d7dee8;
      padding: .45rem .6rem;
      vertical-align: top;
    }
    .obsidian-preview blockquote {
      border-left: 4px solid #94a3b8;
      background: #f8fafc;
      padding: .35rem .8rem;
      margin: .8rem 0;
    }
    .obsidian-preview mjx-container {
      font-size: 108%;
    }
    .obsidian-preview .math-inline {
      display: inline;
      white-space: nowrap;
    }
    .obsidian-preview .math-block {
      display: block;
      overflow-x: auto;
      text-align: center;
      margin: .8rem 0;
    }
    .obsidian-preview math {
      font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
      font-size: 1.02em;
    }
    .obsidian-preview .math-basic {
      color: #111827;
      font-family: "Cambria Math", "Times New Roman", "SimSun", serif;
    }
    .obsidian-preview .math-frac {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      vertical-align: middle;
      line-height: 1.05;
      margin: 0 .12em;
      font-size: .92em;
    }
    .obsidian-preview .math-frac .math-num {
      display: block;
      border-bottom: 1px solid currentColor;
      padding: 0 .18em .08em;
    }
    .obsidian-preview .math-frac .math-den {
      display: block;
      padding: .08em .18em 0;
    }
    .obsidian-preview .math-root {
      display: inline-flex;
      align-items: baseline;
      gap: .05em;
    }
    .obsidian-preview .math-root > span {
      border-top: 1px solid currentColor;
      padding-left: .08em;
    }
    .obsidian-preview .math-overline {
      text-decoration: overline;
    }
    .obsidian-preview .math-vector {
      position: relative;
      display: inline-block;
      padding-top: .12em;
    }
    .obsidian-preview .math-vector::before {
      content: "→";
      position: absolute;
      left: 0;
      right: 0;
      top: -.75em;
      text-align: center;
      font-size: .75em;
    }
    .obsidian-preview .math-sum {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      vertical-align: middle;
      line-height: 1;
      margin: 0 .15em;
    }
    .obsidian-preview .math-sum-symbol {
      font-size: 1.35em;
      line-height: .9;
    }
    .obsidian-preview .math-sum-upper,
    .obsidian-preview .math-sum-lower {
      font-size: .65em;
      line-height: .9;
      min-height: .8em;
    }
    .obsidian-preview sup,
    .obsidian-preview sub {
      line-height: 0;
      font-size: 75%;
    }
    .obsidian-preview .obsidian-wikilink {
      color: #6d28d9;
      text-decoration: none;
      border-bottom: 1px solid rgba(109, 40, 217, .35);
    }
    .obsidian-preview mark {
      background: #fff3a3;
      color: inherit;
      padding: 0 .12rem;
      border-radius: 2px;
    }
    @media print {
      .print-preview-button {
        display: none;
      }
      .obsidian-preview {
        max-width: none;
        padding: 0;
        margin: 0;
      }
      @page {
        size: A4;
        margin: 12mm;
      }
    }
    """


def _convert_obsidian_markup_to_markdown(markdown_text: str) -> str:
    """Handle a small useful subset of Obsidian-only Markdown syntax."""

    text = re.sub(
        r"!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
        lambda match: f"![{match.group(2) or match.group(1)}]({match.group(1)})",
        markdown_text,
    )
    text = re.sub(
        r"\[\[([^\]|]+)\|([^\]]+)\]\]",
        r'<span class="obsidian-wikilink">\2</span>',
        text,
    )
    text = re.sub(
        r"\[\[([^\]]+)\]\]",
        r'<span class="obsidian-wikilink">\1</span>',
        text,
    )
    text = re.sub(r"==(.+?)==", r"<mark>\1</mark>", text)
    return text


def _embed_local_media_links(
    markdown_text: str,
    *,
    kb_root: Path,
    markdown_relative_path: str,
) -> str:
    """Convert local Markdown image references to data URIs for iframe preview."""

    markdown_dir = (kb_root / markdown_relative_path).resolve().parent

    def replace_markdown_image(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        original_target = match.group(2).strip()
        media_path = _resolve_media_reference(
            original_target,
            kb_root=kb_root,
            markdown_dir=markdown_dir,
        )
        if not media_path:
            return match.group(0)

        data_uri = _image_file_to_data_uri(media_path)
        if not data_uri:
            return match.group(0)
        return f"![{alt_text}]({data_uri})"

    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        replace_markdown_image,
        markdown_text,
    )

    def replace_html_img(match: re.Match[str]) -> str:
        prefix = match.group(1)
        quote = match.group(2)
        original_target = match.group(3).strip()
        suffix = match.group(4)
        media_path = _resolve_media_reference(
            original_target,
            kb_root=kb_root,
            markdown_dir=markdown_dir,
        )
        if not media_path:
            return match.group(0)
        data_uri = _image_file_to_data_uri(media_path)
        if not data_uri:
            return match.group(0)
        return f"{prefix}{quote}{data_uri}{suffix}"

    return re.sub(
        r"(<img[^>]+src=)([\"'])([^\"']+)([\"'][^>]*>)",
        replace_html_img,
        text,
    )


def _resolve_media_reference(
    reference: str,
    *,
    kb_root: Path,
    markdown_dir: Path,
) -> Path | None:
    reference = reference.strip().strip("<>").replace("\\", "/")
    if not reference or reference.startswith(("http://", "https://", "data:")):
        return None

    candidates: list[Path] = []
    ref_path = Path(reference)
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        candidates.extend(
            [
                markdown_dir / ref_path,
                kb_root / ref_path,
                _media_root(kb_root) / ref_path,
                _media_root(kb_root) / ref_path.name,
                kb_root / "_raw" / ref_path,
                kb_root / "_raw" / ref_path.name,
            ]
        )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file() and _is_image_file(resolved):
            return resolved
    return None


def _image_file_to_data_uri(path: Path) -> str | None:
    if not _is_image_file(path):
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime_type};base64,{encoded}"


def _convert_latex_math_to_html(markdown_text: str) -> str:
    """Render Obsidian-style LaTeX math delimiters into MathML HTML."""

    protected_blocks: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        protected_blocks.append(match.group(0))
        return f"@@CODE_BLOCK_{len(protected_blocks) - 1}@@"

    text = re.sub(r"```.*?```", protect_code, markdown_text, flags=re.DOTALL)

    patterns = [
        (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), True),
        (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), True),
        (re.compile(r"\\\((.+?)\\\)"), False),
        (re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"), False),
    ]
    for pattern, display in patterns:
        text = pattern.sub(
            lambda match, is_display=display: _latex_to_mathml(
                match.group(1),
                display=is_display,
            ),
            text,
        )

    for index, block in enumerate(protected_blocks):
        text = text.replace(f"@@CODE_BLOCK_{index}@@", block)
    return text


def _latex_to_mathml(latex: str, *, display: bool) -> str:
    latex = latex.strip()
    if not latex:
        return ""

    try:
        from latex2mathml.converter import convert

        mathml = convert(latex, display=display)
        class_name = "math-block" if display else "math-inline"
        return f'<span class="{class_name}">{mathml}</span>'
    except Exception:
        rendered = _latex_to_basic_html(latex)
        if display:
            return f'<span class="math-block math-basic">{rendered}</span>'
        return f'<span class="math-inline math-basic">{rendered}</span>'


def _latex_to_basic_html(latex: str) -> str:
    """Best-effort local renderer for common Gaokao LaTeX snippets."""

    expr = latex.strip()
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = _render_latex_commands(expr)
    expr = _replace_latex_symbols(expr)
    expr = _render_scripts(expr)
    return expr


def _render_latex_commands(expr: str) -> str:
    expr = _replace_frac(expr)
    expr = _replace_one_arg_command(
        expr,
        "sqrt",
        lambda value: f'<span class="math-root">√<span>{_latex_to_basic_html(value)}</span></span>',
    )
    expr = _replace_one_arg_command(
        expr,
        "bar",
        lambda value: f'<span class="math-overline">{_latex_to_basic_html(value)}</span>',
    )
    expr = _replace_one_arg_command(
        expr,
        "vec",
        lambda value: f'<span class="math-vector">{_latex_to_basic_html(value)}</span>',
    )
    return (
        html.escape(expr, quote=False)
        .replace("&lt;span", "<span")
        .replace("&lt;/span&gt;", "</span>")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )


def _replace_frac(expr: str) -> str:
    command = "\\frac"
    while command in expr:
        start = expr.find(command)
        numerator_start = start + len(command)
        numerator, numerator_end = _consume_latex_group(expr, numerator_start)
        if numerator is None:
            break
        denominator, denominator_end = _consume_latex_group(expr, numerator_end)
        if denominator is None:
            break
        rendered = (
            '<span class="math-frac">'
            f'<span class="math-num">{_latex_to_basic_html(numerator)}</span>'
            f'<span class="math-den">{_latex_to_basic_html(denominator)}</span>'
            "</span>"
        )
        expr = expr[:start] + rendered + expr[denominator_end:]
    return expr


def _replace_one_arg_command(expr: str, command_name: str, renderer) -> str:
    command = f"\\{command_name}"
    while command in expr:
        start = expr.find(command)
        group_start = start + len(command)
        value, group_end = _consume_latex_group(expr, group_start)
        if value is None:
            break
        expr = expr[:start] + renderer(value) + expr[group_end:]
    return expr


def _consume_latex_group(expr: str, start: int) -> tuple[str | None, int]:
    while start < len(expr) and expr[start].isspace():
        start += 1
    if start >= len(expr) or expr[start] != "{":
        return None, start

    depth = 0
    for index in range(start, len(expr)):
        char = expr[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return expr[start + 1:index], index + 1
    return None, start


def _replace_latex_symbols(expr: str) -> str:
    replacements = {
        "\\cap": "∩",
        "\\cup": "∪",
        "\\in": "∈",
        "\\notin": "∉",
        "\\subset": "⊂",
        "\\subseteq": "⊆",
        "\\le": "≤",
        "\\leq": "≤",
        "\\ge": "≥",
        "\\geq": "≥",
        "\\ne": "≠",
        "\\neq": "≠",
        "\\pm": "±",
        "\\times": "×",
        "\\cdot": "·",
        "\\circ": "°",
        "\\perp": "⊥",
        "\\parallel": "∥",
        "\\angle": "∠",
        "\\triangle": "△",
        "\\lambda": "λ",
        "\\mu": "μ",
        "\\pi": "π",
        "\\alpha": "α",
        "\\beta": "β",
        "\\gamma": "γ",
        "\\sin": "sin",
        "\\cos": "cos",
        "\\tan": "tan",
        "\\ln": "ln",
        "\\log": "log",
        "\\infty": "∞",
        "\\{": "{",
        "\\}": "}",
    }
    for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
        expr = expr.replace(source, target)
    return expr


def _render_scripts(expr: str) -> str:
    expr = re.sub(
        r"\^\{([^{}]+)\}",
        lambda match: f"<sup>{html.escape(_replace_latex_symbols(match.group(1)))}</sup>",
        expr,
    )
    expr = re.sub(
        r"_\{([^{}]+)\}",
        lambda match: f"<sub>{html.escape(_replace_latex_symbols(match.group(1)))}</sub>",
        expr,
    )
    expr = re.sub(r"\^([A-Za-z0-9+\-]+)", r"<sup>\1</sup>", expr)
    expr = re.sub(r"_([A-Za-z0-9+\-]+)", r"<sub>\1</sub>", expr)
    return expr


def _list_raw_files(kb_root: Path) -> list[str]:
    raw_root = kb_root / "_raw"
    if not raw_root.exists():
        return []
    files = [
        str(path.relative_to(raw_root))
        for path in raw_root.rglob("*")
        if path.is_file()
    ]
    return sorted(files)


def _deduplicate_raw_files(raw_root: Path) -> tuple[int, int]:
    if not raw_root.exists():
        return 0, 0

    by_hash: dict[str, list[Path]] = {}
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            digest = _file_digest(path)
        except OSError:
            continue
        by_hash.setdefault(digest, []).append(path)

    deleted_count = 0
    freed_bytes = 0
    for duplicates in by_hash.values():
        if len(duplicates) < 2:
            continue
        keep, *remove = sorted(duplicates, key=_raw_keep_priority)
        del keep
        for path in remove:
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            deleted_count += 1
            freed_bytes += size

    return deleted_count, freed_bytes


def _file_digest(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _raw_keep_priority(path: Path) -> tuple[int, int, str]:
    parts = set(path.parts)
    if "web" in parts or "uploads" in parts:
        priority = 0
    elif any(part.startswith("_") for part in parts):
        priority = 2
    else:
        priority = 1
    return priority, len(path.parts), str(path)


def _media_root(kb_root: Path) -> Path:
    return kb_root / "assets"


def _list_media_files(kb_root: Path) -> list[str]:
    media_root = _media_root(kb_root)
    if not media_root.exists():
        return []
    files = [
        str(path.relative_to(kb_root))
        for path in media_root.rglob("*")
        if path.is_file() and _is_image_file(path)
    ]
    return sorted(files)


def _unique_media_path(media_root: Path, filename: str) -> Path:
    safe_name = re.sub(r'[<>:"/\\|?*\s]+', "_", Path(filename).name)
    if not safe_name:
        safe_name = "media.png"

    target = media_root / safe_name
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    index = 2
    while True:
        candidate = media_root / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _is_image_file(path: Path) -> bool:
    mime_type = mimetypes.guess_type(path.name)[0] or ""
    return mime_type.startswith("image/")


def _safe_child_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    target = (resolved_root / relative_path).resolve()
    if resolved_root != target and resolved_root not in target.parents:
        raise ValueError("Path escapes the target root.")
    return target


def _format_file_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size} B"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fb;
        }
        h1, h2, h3 {
            color: #1f2937;
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] {
            background: #eef3f8;
            border-right: 1px solid #d7dee8;
        }
        div[data-testid="stForm"] {
            border: 1px solid #dde3ea;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 8px;
            border-color: #dfe6ee;
            background: #ffffff;
        }
        .stButton button, .stDownloadButton button, .stLinkButton a {
            border-radius: 6px;
            font-weight: 600;
        }
        .obsidian-preview {
            background: #ffffff;
            border: 1px solid #dfe6ee;
            border-radius: 8px;
            padding: 2rem 3rem;
            color: #1f2937;
            line-height: 1.9;
            font-size: 18px;
            max-width: 1040px;
            margin: 0 auto;
            overflow-x: auto;
            font-family: "Times New Roman", "Noto Serif CJK SC", "SimSun", serif;
        }
        .obsidian-preview p {
            margin: .65rem 0 1rem;
        }
        .obsidian-preview h1,
        .obsidian-preview h2,
        .obsidian-preview h3 {
            border-bottom: 0;
            padding-bottom: 0;
            margin: 1.35rem 0 .8rem;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        }
        .obsidian-preview h1 { font-size: 1.65rem; }
        .obsidian-preview h2 { font-size: 1.35rem; }
        .obsidian-preview h3 { font-size: 1.15rem; }
        .obsidian-preview table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .obsidian-preview th,
        .obsidian-preview td {
            border: 1px solid #d7dee8;
            padding: .45rem .6rem;
            vertical-align: top;
        }
        .obsidian-preview blockquote {
            border-left: 4px solid #94a3b8;
            background: #f8fafc;
            padding: .35rem .8rem;
            margin: .8rem 0;
        }
        .obsidian-preview .math-inline {
            display: inline;
            white-space: nowrap;
        }
        .obsidian-preview .math-block {
            display: block;
            overflow-x: auto;
            text-align: center;
            margin: .8rem 0;
        }
        .obsidian-preview math {
            font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
            font-size: 1.02em;
        }
        .obsidian-preview .math-fallback {
            color: #7c3aed;
            font-family: "Consolas", monospace;
        }
        .obsidian-preview .math-basic {
            color: #111827;
            font-family: "Cambria Math", "Times New Roman", "SimSun", serif;
        }
        .obsidian-preview .math-frac {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            vertical-align: middle;
            line-height: 1.05;
            margin: 0 .12em;
            font-size: .92em;
        }
        .obsidian-preview .math-frac .math-num {
            display: block;
            border-bottom: 1px solid currentColor;
            padding: 0 .18em .08em;
        }
        .obsidian-preview .math-frac .math-den {
            display: block;
            padding: .08em .18em 0;
        }
        .obsidian-preview .math-root {
            display: inline-flex;
            align-items: baseline;
            gap: .05em;
        }
        .obsidian-preview .math-root > span {
            border-top: 1px solid currentColor;
            padding-left: .08em;
        }
        .obsidian-preview .math-overline {
            text-decoration: overline;
        }
        .obsidian-preview .math-vector {
            position: relative;
            display: inline-block;
            padding-top: .12em;
        }
    .obsidian-preview .math-vector::before {
      content: "→";
      position: absolute;
            left: 0;
            right: 0;
            top: -.75em;
      text-align: center;
      font-size: .75em;
    }
    .obsidian-preview .math-sum {
      display: inline-flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      vertical-align: middle;
      line-height: 1;
      margin: 0 .15em;
    }
    .obsidian-preview .math-sum-symbol {
      font-size: 1.35em;
      line-height: .9;
    }
    .obsidian-preview .math-sum-upper,
    .obsidian-preview .math-sum-lower {
      font-size: .65em;
      line-height: .9;
      min-height: .8em;
    }
    .obsidian-preview .obsidian-wikilink {
            color: #6d28d9;
            text-decoration: none;
            border-bottom: 1px solid rgba(109, 40, 217, .35);
        }
        .obsidian-preview mark {
            background: #fff3a3;
            color: inherit;
            padding: 0 .12rem;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
