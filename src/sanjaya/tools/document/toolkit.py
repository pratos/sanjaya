"""DocumentToolkit — text document analysis toolkit for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...answer import Evidence
from ...retrieval.sqlite_fts import SQLiteFTSBackend
from ..base import Tool, Toolkit, ToolParam
from .parsers import DocumentChunk, parse_document


_DOCUMENT_STRATEGY_PROMPT = """\
## Available Tools

You are analyzing text documents. You decide the strategy.

### Document tools:

- **list_documents()** — see all loaded documents with page/section counts.
- **search_documents(query, top_k=10)** — BM25 keyword search across all documents.
  Returns matching chunks with scores, doc_id, and chunk labels.
- **read_chunk(doc_id, chunk_index)** — read the full text of a specific chunk
  (page, slide, section, or paragraph). Use after search to get full context.
- **get_document_info(doc_id)** — detailed info about a single document.

### Text tools (builtins):

- **llm_query(prompt)** — text-only reasoning. Feed it search results
  and chunk text for analysis and synthesis. Cheap, sees everything at once.

### Strategy:

- Start with list_documents() to see what you have.
- Use search_documents() to find relevant sections by keyword.
- Use read_chunk() to get full text of interesting hits.
- Feed accumulated content to llm_query() for synthesis.
- Always: explore before answering, cite document sources, print results.

### Critical: ground everything in the text

- **Domain vocabulary is fine** — use standard terminology from the question's
  domain (legal terms, financial metrics, technical jargon) to formulate
  search queries. This is expected.
- **Content assumptions are not** — do NOT assume what the document says.
  Do not pre-populate answers, names, figures, or conclusions from your
  training data. Every factual claim must come from a chunk you read.
- Start with terms from the question. Read the results. Use what you
  *actually found* to formulate follow-up queries.
- If the question asks "who", "what", or "how much", the answer must emerge
  from the documents — do not seed your search with specific answers you
  already know from training.

### One code block per response. Print everything.
"""


@dataclass
class DocumentInfo:
    """Metadata about a loaded document."""

    doc_id: str
    file_path: str
    file_type: str
    num_chunks: int
    chunk_type: str  # "pages", "slides", "sections", "paragraphs"
    total_chars: int


_CHUNK_TYPE_MAP = {
    "pdf": "pages",
    "pptx": "slides",
    "md": "sections",
    "epub": "chapters",
    "txt": "paragraphs",
}


class DocumentToolkit(Toolkit):
    """Text document analysis toolkit with search, reading, and introspection tools."""

    def __init__(self) -> None:
        self._documents: dict[str, DocumentInfo] = {}
        self._chunks: dict[str, list[DocumentChunk]] = {}
        self._all_chunks: list[DocumentChunk] = []
        self._fts: SQLiteFTSBackend | None = None
        self._accessed_chunks: set[tuple[str, int]] = set()
        self._question: str | None = None

        # Injected by Agent.use()
        self._llm_client: Any = None
        self._tracer: Any = None
        self._budget: Any = None

    def setup(self, context: dict[str, Any]) -> None:
        """Parse and index all documents."""
        doc_paths = context.get("document")
        if not doc_paths:
            return

        self._question = context.get("question")

        if isinstance(doc_paths, str):
            doc_paths = [doc_paths]

        existing_ids: set[str] = set()
        all_chunks: list[DocumentChunk] = []

        for path in doc_paths:
            chunks = parse_document(path, existing_ids=existing_ids)
            if not chunks:
                continue

            doc_id = chunks[0].doc_id
            existing_ids.add(doc_id)
            self._chunks[doc_id] = chunks
            all_chunks.extend(chunks)

            file_type = chunks[0].metadata.get("file_type", "txt")
            self._documents[doc_id] = DocumentInfo(
                doc_id=doc_id,
                file_path=chunks[0].metadata.get("file_path", path),
                file_type=file_type,
                num_chunks=len(chunks),
                chunk_type=_CHUNK_TYPE_MAP.get(file_type, "chunks"),
                total_chars=sum(len(c.text) for c in chunks),
            )

        self._all_chunks = all_chunks

        if all_chunks:
            self._fts = SQLiteFTSBackend(path=":memory:")
            self._fts.index(
                documents=[c.text for c in all_chunks],
                metadata=[c.metadata for c in all_chunks],
                collection="documents",
            )

    def teardown(self) -> None:
        pass

    def tools(self) -> list[Tool]:
        return [
            self._make_list_documents_tool(),
            self._make_search_documents_tool(),
            self._make_read_chunk_tool(),
            self._make_get_document_info_tool(),
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "documents_loaded": len(self._documents),
            "total_chunks": len(self._all_chunks),
            "chunks_accessed": len(self._accessed_chunks),
            "documents": {
                doc_id: {
                    "file_type": info.file_type,
                    "num_chunks": info.num_chunks,
                    "chunk_type": info.chunk_type,
                    "total_chars": info.total_chars,
                }
                for doc_id, info in self._documents.items()
            },
        }

    def build_evidence(self) -> list[Evidence]:
        by_doc: dict[str, list[int]] = {}
        for doc_id, chunk_index in self._accessed_chunks:
            by_doc.setdefault(doc_id, []).append(chunk_index)

        evidence: list[Evidence] = []
        for doc_id, indices in by_doc.items():
            info = self._documents.get(doc_id)
            if not info:
                continue
            sorted_indices = sorted(indices)
            labels = []
            for idx in sorted_indices:
                chunks = self._chunks.get(doc_id, [])
                for c in chunks:
                    if c.chunk_index == idx:
                        labels.append(c.chunk_label)
                        break
            evidence.append(
                Evidence(
                    source=f"document:{doc_id}",
                    rationale=f"Accessed {len(sorted_indices)} {info.chunk_type} from {info.file_path}: {', '.join(labels)}",
                    artifacts={
                        "doc_id": doc_id,
                        "file_path": info.file_path,
                        "chunks_accessed": sorted_indices,
                    },
                )
            )
        return evidence

    def prompt_section(self) -> str | None:
        if not self._documents:
            return None

        parts = [_DOCUMENT_STRATEGY_PROMPT]

        doc_summaries = []
        for info in self._documents.values():
            doc_summaries.append(f"{info.doc_id} ({info.num_chunks} {info.chunk_type})")
        parts.append(f"\n{len(self._documents)} document(s) loaded: {', '.join(doc_summaries)}")

        return "\n".join(parts)

    # ── Tool factories ──────────────────────────────────────

    def _make_list_documents_tool(self) -> Tool:
        toolkit = self

        def _list_documents() -> list[dict]:
            """List all loaded documents with their chunk counts."""
            return [
                {
                    "doc_id": info.doc_id,
                    "file_type": info.file_type,
                    "num_chunks": info.num_chunks,
                    "chunk_type": info.chunk_type,
                    "total_chars": info.total_chars,
                }
                for info in toolkit._documents.values()
            ]

        return Tool(
            name="list_documents",
            description="List all loaded documents with type, chunk count, and size.",
            fn=_list_documents,
            parameters={},
            return_type="list[dict]",
        )

    def _make_search_documents_tool(self) -> Tool:
        toolkit = self

        def _search_documents(query: str, top_k: int = 10) -> list[dict]:
            """Search across all documents by keyword/phrase.

            Returns ranked results with text snippets, doc_id, chunk labels, and scores.
            """
            if toolkit._fts is None:
                return [{"error": "No documents indexed"}]

            results = toolkit._fts.search(query, top_k=top_k, collection="documents")
            out = []
            for r in results:
                meta = r.get("metadata", {})
                doc_id = meta.get("doc_id", "unknown")
                chunk_index = meta.get("chunk_index", -1)
                toolkit._accessed_chunks.add((doc_id, chunk_index))

                text = r["text"]
                snippet = text[:500] + "..." if len(text) > 500 else text
                out.append(
                    {
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "chunk_label": meta.get("chunk_label", ""),
                        "text": snippet,
                        "score": round(r["score"], 4),
                    }
                )
            return out

        return Tool(
            name="search_documents",
            description="Search all loaded documents by keyword/phrase. Returns matching chunks with doc_id, labels, and relevance scores.",
            fn=_search_documents,
            parameters={
                "query": ToolParam(name="query", type_hint="str", description="Search query."),
                "top_k": ToolParam(name="top_k", type_hint="int", default=10, description="Max results."),
            },
            return_type="list[dict]",
        )

    def _make_read_chunk_tool(self) -> Tool:
        toolkit = self
        max_chars = 20_000

        def _read_chunk(doc_id: str, chunk_index: int) -> dict:
            """Read the text of a specific chunk (page, slide, section, or paragraph).

            Use after search_documents() to get the content of a matching chunk.
            Large chunks are capped at 20K chars — use search_documents() to
            find the relevant passages within them.
            """
            chunks = toolkit._chunks.get(doc_id)
            if chunks is None:
                return {"error": f"Unknown doc_id: {doc_id}. Use list_documents() to see available documents."}

            for c in chunks:
                if c.chunk_index == chunk_index:
                    toolkit._accessed_chunks.add((doc_id, chunk_index))
                    text = c.text
                    truncated = False
                    if len(text) > max_chars:
                        text = text[:max_chars]
                        truncated = True
                    result: dict = {
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "chunk_label": c.chunk_label,
                        "text": text,
                        "total_chars": len(c.text),
                    }
                    if truncated:
                        result["truncated"] = True
                        result["hint"] = (
                            f"Chunk is {len(c.text)} chars, showing first {max_chars}. "
                            "Use search_documents() with specific keywords to find "
                            "relevant passages within this chunk."
                        )
                    return result

            return {
                "error": f"chunk_index {chunk_index} not found in {doc_id}. Valid range: 0-{len(chunks) - 1}."
            }

        return Tool(
            name="read_chunk",
            description="Read the text of a specific chunk (page, slide, section, paragraph) by doc_id and chunk_index. Large chunks are capped at 20K chars.",
            fn=_read_chunk,
            parameters={
                "doc_id": ToolParam(name="doc_id", type_hint="str", description="Document ID from list_documents()."),
                "chunk_index": ToolParam(name="chunk_index", type_hint="int", description="Chunk index from search_documents() or get_document_info()."),
            },
            return_type="dict",
        )

    def _make_get_document_info_tool(self) -> Tool:
        toolkit = self

        def _get_document_info(doc_id: str) -> dict:
            """Get detailed info about a document: path, type, and all chunk labels."""
            info = toolkit._documents.get(doc_id)
            if info is None:
                return {"error": f"Unknown doc_id: {doc_id}. Use list_documents() to see available documents."}

            chunks = toolkit._chunks.get(doc_id, [])
            return {
                "doc_id": info.doc_id,
                "file_path": info.file_path,
                "file_type": info.file_type,
                "num_chunks": info.num_chunks,
                "chunk_type": info.chunk_type,
                "total_chars": info.total_chars,
                "chunks": [
                    {"index": c.chunk_index, "label": c.chunk_label, "chars": len(c.text)}
                    for c in chunks
                ],
            }

        return Tool(
            name="get_document_info",
            description="Get detailed info about a document including its structure and all chunk labels.",
            fn=_get_document_info,
            parameters={
                "doc_id": ToolParam(name="doc_id", type_hint="str", description="Document ID from list_documents()."),
            },
            return_type="dict",
        )
