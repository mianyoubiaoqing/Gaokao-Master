"""Core tools for the Gaokao-Master agent system."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from loguru import logger
from rank_bm25 import BM25Okapi

from gaokao_master.kb.manager import (
    DocumentExtractionError,
    KnowledgeBaseManager,
    SUPPORTED_EXTENSIONS,
)


WorkspaceAction = Literal["read", "write", "update", "delete", "list"]


@dataclass(frozen=True)
class RetrievalHit:
    """A single hybrid retrieval result."""

    text: str
    score: float
    source: str
    subject: str
    topic: str
    chunk_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScrapedResource:
    """A downloaded web resource and its ingestion status."""

    title: str
    url: str
    local_path: Path
    ingested_markdown_path: Path | None
    status: str


def fuzzy_retrieve(
    query: str,
    kb_manager: KnowledgeBaseManager | None = None,
    *,
    kb_root: str | Path = "Gaokao_KB",
    subject: str | None = None,
    topic: str | None = None,
    top_k: int = 8,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[RetrievalHit]:
    """Perform hybrid retrieval over the local KB.

    It combines ChromaDB semantic search with local BM25 keyword search. The
    function can be wrapped as a LangChain/LangGraph tool later without changing
    its signature much.
    """

    if not query.strip():
        return []

    manager = kb_manager or KnowledgeBaseManager(kb_root=kb_root)
    semantic_hits = _semantic_search(manager, query, subject, topic, top_k * 2)
    keyword_hits = _keyword_search(manager, query, subject, topic, top_k * 2)

    merged: dict[str, RetrievalHit] = {}
    for hit in semantic_hits:
        merged[hit.chunk_id] = hit

    for hit in keyword_hits:
        existing = merged.get(hit.chunk_id)
        if existing is None:
            merged[hit.chunk_id] = hit
            continue

        merged[hit.chunk_id] = RetrievalHit(
            text=existing.text,
            score=min(
                1.0,
                existing.score * semantic_weight + hit.score * keyword_weight,
            ),
            source=existing.source,
            subject=existing.subject,
            topic=existing.topic,
            chunk_id=existing.chunk_id,
            metadata=existing.metadata,
        )

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]


def workspace_editor(
    action: WorkspaceAction,
    relative_path: str | None = None,
    *,
    content: str | None = None,
    old_text: str | None = None,
    new_text: str | None = None,
    kb_root: str | Path = "Gaokao_KB",
) -> dict[str, Any]:
    """Read, write, update, delete, or list Markdown files inside the KB.

    All paths are constrained to the KB root to keep workspace edits safe.
    """

    root = Path(kb_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if action == "list":
        files = [
            str(path.relative_to(root))
            for path in root.rglob("*.md")
            if not any(part.startswith(".") or part == "_raw" for part in path.parts)
        ]
        return {"action": action, "files": sorted(files)}

    if not relative_path:
        raise ValueError("relative_path is required for this workspace action.")

    target = _safe_workspace_path(root, relative_path)

    if action == "read":
        return {
            "action": action,
            "path": str(target),
            "content": target.read_text(encoding="utf-8"),
        }

    if action == "write":
        if content is None:
            raise ValueError("content is required for write.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"action": action, "path": str(target), "bytes": target.stat().st_size}

    if action == "update":
        if old_text is None or new_text is None:
            raise ValueError("old_text and new_text are required for update.")
        current = target.read_text(encoding="utf-8")
        if old_text not in current:
            raise ValueError("old_text was not found in the target file.")
        updated = current.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return {"action": action, "path": str(target), "replacements": 1}

    if action == "delete":
        target.unlink()
        return {"action": action, "path": str(target), "deleted": True}

    raise ValueError(f"Unsupported workspace action: {action}")


def web_resource_scraper(
    query: str,
    subject: str,
    topic: str,
    *,
    kb_manager: KnowledgeBaseManager | None = None,
    kb_root: str | Path = "Gaokao_KB",
    max_results: int = 5,
    timeout: int = 20,
    resource_urls: list[str] | None = None,
    tavily_api_key: str | None = None,
) -> list[ScrapedResource]:
    """Search and download Gaokao resources, then ingest supported files.

    PDF and DOCX files are parsed into Markdown and indexed. Legacy `.doc`
    files are downloaded but left unparsed because reliable local conversion is
    platform-dependent. If the search engine is unavailable, the function
    returns whatever direct URL downloads succeeded instead of raising into UI.
    """

    manager = kb_manager or KnowledgeBaseManager(kb_root=kb_root)
    download_dir = (
        manager.paths.raw
        / "web"
        / manager._safe_part(subject)
        / manager._safe_part(topic)
    )
    download_dir.mkdir(parents=True, exist_ok=True)

    resources: list[ScrapedResource] = []
    direct_urls = [url.strip() for url in resource_urls or [] if url.strip()]
    for direct_url in direct_urls:
        if len(resources) >= max_results:
            break
        for candidate_url in _discover_download_urls(direct_url, timeout=timeout):
            if len(resources) >= max_results:
                break
            resource = _download_and_ingest(
                url=candidate_url,
                title=Path(urlparse(candidate_url).path).stem or "Gaokao resource",
                subject=subject,
                topic=topic,
                manager=manager,
                download_dir=download_dir,
                timeout=timeout,
            )
            if resource:
                resources.append(resource)

    if len(resources) >= max_results or not query.strip():
        return resources

    search_query = (
        f"{query} 高考 试卷 试题 答案 解析 PDF DOCX "
        "-志愿 -分数线 -录取 -招生 -大学排名"
    )

    results = _search_resource_pages(
        search_query,
        max_results=max_results * 3,
        timeout=timeout,
        tavily_api_key=tavily_api_key,
    )

    for result in results:
        if len(resources) >= max_results:
            break

        title = result.get("title") or "Gaokao resource"
        url = result.get("href") or result.get("url")
        if not url:
            continue
        if not _looks_like_exam_search_result(title, url, subject, topic):
            logger.info("Skipped non-exam search result: {} {}", title, url)
            continue

        candidate_urls = _discover_download_urls(url, timeout=timeout)
        for candidate_url in candidate_urls:
            if len(resources) >= max_results:
                break
            resource = _download_and_ingest(
                url=candidate_url,
                title=title,
                subject=subject,
                topic=topic,
                manager=manager,
                download_dir=download_dir,
                timeout=timeout,
            )
            if resource:
                resources.append(resource)

    return resources


def _search_resource_pages(
    query: str,
    *,
    max_results: int,
    timeout: int,
    tavily_api_key: str | None = None,
) -> list[dict[str, str]]:
    """Search resource pages through several best-effort providers."""

    merged: dict[str, dict[str, str]] = {}

    search_batches = [
        _search_with_tavily(
            query,
            max_results=max_results,
            timeout=timeout,
            tavily_api_key=tavily_api_key,
        )
    ]
    if len(search_batches[0]) < max_results:
        search_batches.append(
            _search_with_duckduckgo(query, max_results=max_results, timeout=timeout)
        )
    if sum(len(batch) for batch in search_batches) < max_results:
        search_batches.append(
            _search_with_bing_html(query, max_results=max_results, timeout=timeout)
        )
    if sum(len(batch) for batch in search_batches) < max_results:
        search_batches.append(
            _search_with_sogou_html(query, max_results=max_results, timeout=timeout)
        )

    for batch in search_batches:
        for item in batch:
            if len(merged) >= max_results:
                break
            url = item.get("href") or item.get("url")
            if url and url not in merged:
                merged[url] = {
                    "title": item.get("title") or "Gaokao resource",
                    "href": url,
                }

    return list(merged.values())[:max_results]


def _search_with_tavily(
    query: str,
    *,
    max_results: int,
    timeout: int,
    tavily_api_key: str | None = None,
) -> list[dict[str, str]]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
            },
            timeout=timeout,
            headers=_http_headers(),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Tavily search unavailable: {}", exc)
        return []

    items = response.json().get("results", [])
    return [
        {
            "title": item.get("title") or "Gaokao resource",
            "href": item.get("url", ""),
        }
        for item in items
        if item.get("url")
    ]


def _search_with_duckduckgo(
    query: str,
    *,
    max_results: int,
    timeout: int,
) -> list[dict[str, str]]:
    del timeout
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.info("DuckDuckGo search unavailable: {}", exc)
        return []


def _search_with_bing_html(
    query: str,
    *,
    max_results: int,
    timeout: int,
) -> list[dict[str, str]]:
    try:
        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query},
            timeout=timeout,
            headers=_http_headers(),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Bing HTML search unavailable: {}", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    for anchor in soup.select("li.b_algo h2 a[href], h2 a[href], a[href]"):
        href = anchor.get("href", "")
        if not href.startswith(("http://", "https://")):
            continue
        results.append({"title": anchor.get_text(" ", strip=True), "href": href})
        if len(results) >= max_results:
            break
    return results


def _search_with_sogou_html(
    query: str,
    *,
    max_results: int,
    timeout: int,
) -> list[dict[str, str]]:
    try:
        response = requests.get(
            "https://www.sogou.com/web",
            params={"query": query},
            timeout=timeout,
            headers=_http_headers(),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Sogou HTML search unavailable: {}", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        if not href.startswith(("http://", "https://")):
            continue
        title = anchor.get_text(" ", strip=True)
        if not title:
            continue
        results.append({"title": title, "href": href})
        if len(results) >= max_results:
            break
    return results


def _semantic_search(
    manager: KnowledgeBaseManager,
    query: str,
    subject: str | None,
    topic: str | None,
    limit: int,
) -> list[RetrievalHit]:
    filters = _build_chroma_filter(subject, topic)
    kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if filters:
        kwargs["where"] = filters

    try:
        result = manager.collection.query(**kwargs)
    except Exception as exc:  # Chroma raises broad exceptions for empty indexes.
        logger.warning("Semantic search failed: {}", exc)
        return []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    hits: list[RetrievalHit] = []
    for document, metadata, distance, chunk_id in zip(
        documents,
        metadatas,
        distances,
        ids,
        strict=False,
    ):
        score = 1.0 / (1.0 + float(distance or 0.0))
        hits.append(
            RetrievalHit(
                text=document,
                score=score,
                source=metadata.get("markdown_path", ""),
                subject=metadata.get("subject", ""),
                topic=metadata.get("topic", ""),
                chunk_id=chunk_id,
                metadata=metadata,
            )
        )
    return hits


def _keyword_search(
    manager: KnowledgeBaseManager,
    query: str,
    subject: str | None,
    topic: str | None,
    limit: int,
) -> list[RetrievalHit]:
    records: list[tuple[str, dict[str, Any], str]] = []
    for markdown_path in _iter_markdown_files(manager, subject, topic):
        text = markdown_path.read_text(encoding="utf-8", errors="ignore")
        relative = markdown_path.relative_to(manager.paths.root)
        chunks = manager.chunk_markdown(text)
        for index, chunk in enumerate(chunks):
            chunk_id = f"keyword:{markdown_path}:{index}"
            metadata = {
                "markdown_path": str(markdown_path),
                "subject": relative.parts[0] if len(relative.parts) > 0 else "",
                "topic": relative.parts[1] if len(relative.parts) > 1 else "",
                "chunk_index": index,
            }
            records.append((chunk, metadata, chunk_id))

    if not records:
        return []

    tokenized_corpus = [_tokenize_chinese_aware(text) for text, _, _ in records]
    tokenized_query = _tokenize_chinese_aware(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    max_score = max(scores) if len(scores) else 0.0

    ranked_indexes = sorted(
        range(len(records)),
        key=lambda index: scores[index],
        reverse=True,
    )[:limit]

    hits: list[RetrievalHit] = []
    for index in ranked_indexes:
        if scores[index] <= 0:
            continue
        text, metadata, chunk_id = records[index]
        normalized_score = float(scores[index] / max_score) if max_score else 0.0
        hits.append(
            RetrievalHit(
                text=text,
                score=normalized_score,
                source=metadata["markdown_path"],
                subject=metadata["subject"],
                topic=metadata["topic"],
                chunk_id=chunk_id,
                metadata=metadata,
            )
        )

    return hits


def _iter_markdown_files(
    manager: KnowledgeBaseManager,
    subject: str | None,
    topic: str | None,
) -> list[Path]:
    files: list[Path] = []
    for path in manager.paths.root.rglob("*.md"):
        relative = path.relative_to(manager.paths.root)
        if any(part.startswith(".") or part == "_raw" for part in relative.parts):
            continue
        if subject and (len(relative.parts) < 1 or relative.parts[0] != subject):
            continue
        if topic and (len(relative.parts) < 2 or relative.parts[1] != topic):
            continue
        files.append(path)
    return files


def _build_chroma_filter(subject: str | None, topic: str | None) -> dict[str, Any]:
    clauses = []
    if subject:
        clauses.append({"subject": subject})
    if topic:
        clauses.append({"topic": topic})
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _tokenize_chinese_aware(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


def _safe_workspace_path(root: Path, relative_path: str) -> Path:
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise ValueError("Path escapes the knowledge base root.")
    if target.suffix and target.suffix.lower() != ".md":
        raise ValueError("workspace_editor only edits Markdown files.")
    return target


def _discover_download_urls(url: str, *, timeout: int) -> list[str]:
    parsed = urlparse(url)
    if _is_downloadable_path(parsed.path):
        return [url]

    try:
        response = requests.get(url, timeout=timeout, headers=_http_headers())
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch search result page {}: {}", url, exc)
        return []

    content_type = response.headers.get("content-type", "")
    if _looks_like_supported_file(url, content_type):
        return [url]

    soup = BeautifulSoup(response.text, "html.parser")
    candidates: list[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        full_url = urljoin(url, href)
        if _is_downloadable_path(urlparse(full_url).path):
            candidates.append(full_url)

    return list(dict.fromkeys(candidates))


def _download_and_ingest(
    *,
    url: str,
    title: str,
    subject: str,
    topic: str,
    manager: KnowledgeBaseManager,
    download_dir: Path,
    timeout: int,
) -> ScrapedResource | None:
    try:
        response = requests.get(url, timeout=timeout, headers=_http_headers())
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Download failed {}: {}", url, exc)
        return None

    extension = _extension_from_response(url, response.headers.get("content-type", ""))
    if extension not in {".pdf", ".docx", ".doc"}:
        return None

    if not _looks_like_exam_file(
        title=title,
        url=url,
        subject=subject,
        topic=topic,
        content=response.content,
        extension=extension,
    ):
        logger.info("Skipped downloaded non-exam resource: {} {}", title, url)
        return None

    digest = hashlib.sha256(response.content).hexdigest()[:12]
    local_path = download_dir / f"{_slugify(title)}_{digest}{extension}"
    if not local_path.exists():
        local_path.write_bytes(response.content)

    if extension not in SUPPORTED_EXTENSIONS:
        return ScrapedResource(
            title=title,
            url=url,
            local_path=local_path,
            ingested_markdown_path=None,
            status="downloaded_only_legacy_doc",
        )

    try:
        ingested = manager.ingest_file(
            local_path,
            subject=subject,
            topic=topic,
            title=title,
            tags=["web", "mock-exam"],
            source_url=url,
            keep_raw_copy=False,
        )
    except DocumentExtractionError as exc:
        return ScrapedResource(
            title=title,
            url=url,
            local_path=local_path,
            ingested_markdown_path=exc.diagnostic_path,
            status="needs_ocr_or_empty_pdf",
        )
    except Exception as exc:
        logger.warning("Ingestion failed for {}: {}", local_path, exc)
        return ScrapedResource(
            title=title,
            url=url,
            local_path=local_path,
            ingested_markdown_path=None,
            status="ingestion_failed",
        )
    return ScrapedResource(
        title=title,
        url=url,
        local_path=local_path,
        ingested_markdown_path=ingested.markdown_path,
        status="ingested",
    )


def _looks_like_exam_search_result(
    title: str,
    url: str,
    subject: str,
    topic: str,
) -> bool:
    text = f"{title} {url} {subject} {topic}".lower()
    positive = _resource_positive_score(text)
    negative = _resource_negative_score(text)
    return positive >= 2 and negative < 3


def _looks_like_exam_file(
    *,
    title: str,
    url: str,
    subject: str,
    topic: str,
    content: bytes,
    extension: str,
) -> bool:
    metadata_text = f"{title} {url} {subject} {topic}".lower()
    positive = _resource_positive_score(metadata_text)
    negative = _resource_negative_score(metadata_text)

    extracted_text = _extract_resource_preview_text(content, extension).lower()
    if extracted_text:
        positive += _resource_positive_score(extracted_text)
        negative += _resource_negative_score(extracted_text)

    if negative >= 4 and positive < 4:
        return False
    if positive >= 3:
        return True

    # Scanned PDFs often have no text; allow them only when the filename/title
    # already strongly looks like a paper.
    return extension == ".pdf" and _resource_positive_score(metadata_text) >= 3


def _extract_resource_preview_text(content: bytes, extension: str) -> str:
    try:
        if extension == ".pdf":
            import fitz

            texts: list[str] = []
            with fitz.open(stream=content, filetype="pdf") as pdf:
                for page_index in range(min(len(pdf), 2)):
                    page = pdf[page_index]
                    texts.append(page.get_text("text", sort=True))
            return "\n".join(texts)[:5_000]

        if extension == ".docx":
            from docx import Document

            document = Document(BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs[:80]]
            return "\n".join(paragraphs)[:5_000]
    except Exception as exc:
        logger.info("Could not inspect downloaded resource content: {}", exc)
    return ""


def _resource_positive_score(text: str) -> int:
    keywords = [
        "试卷",
        "试题",
        "真题",
        "模拟",
        "一模",
        "二模",
        "三模",
        "联考",
        "押题",
        "答案",
        "解析",
        "高考",
        "pdf",
        "docx",
        "数学",
        "语文",
        "英语",
        "历史",
        "政治",
        "思想政治",
        "地理",
    ]
    return sum(1 for keyword in keywords if keyword.lower() in text)


def _resource_negative_score(text: str) -> int:
    keywords = [
        "志愿",
        "分数线",
        "录取",
        "招生",
        "院校",
        "大学排名",
        "专业排名",
        "简章",
        "政策",
        "新闻",
        "作文素材",
        "范文",
        "知识点",
        "复习方法",
        "广告",
        "会员",
        "付费",
    ]
    return sum(1 for keyword in keywords if keyword.lower() in text)


def _extension_from_response(url: str, content_type: str) -> str:
    parsed_suffix = Path(urlparse(url).path).suffix.lower()
    if parsed_suffix in {".pdf", ".docx", ".doc"}:
        return parsed_suffix
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
    if guessed == ".doc":
        return ".doc"
    if guessed in {".pdf", ".docx"}:
        return guessed
    return ""


def _looks_like_supported_file(url: str, content_type: str) -> bool:
    return _is_downloadable_path(urlparse(url).path) or any(
        marker in content_type.lower()
        for marker in ["pdf", "word", "officedocument"]
    )


def _is_downloadable_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".pdf", ".docx", ".doc"}


def _slugify(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\s]+', "_", value.strip())
    return value[:100] or "gaokao_resource"


def _http_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124 Safari/537.36"
        )
    }
