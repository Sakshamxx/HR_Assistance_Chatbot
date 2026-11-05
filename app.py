#!/usr/bin/env python3
"""
app.py - HR policy assistant: the RAG pipeline.

Retrieval half: loads the Chroma store that ingest.py built and returns the
top-k most relevant policy chunks for a question.
Generation half: sends those chunks to a Groq-hosted model as grounding
context and returns a plain-text answer.

This module holds no web code. server.py imports answer_question() from here
and exposes it over HTTP; this file can also be run on its own:

    venv/bin/python app.py --ask "how much casual leave do I get?"
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Retrieval configuration - these three values MUST match ingest.py exactly.
# A different embedding model produces vectors in a different space, and a
# different collection name points at an empty collection; either mismatch
# fails silently, returning irrelevant results or no results at all.
# ---------------------------------------------------------------------------
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "hr_policies"
EMBEDDING_MODEL = "gemini-embedding-001"

# Number of chunks handed to the LLM as context per question.
TOP_K = 3

# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------
# Groq-hosted. The strongest chat model available on this account; swap for
# "openai/gpt-oss-20b" (faster, cheaper) or "qwen/qwen3.8-27b" if preferred.
ANSWER_MODEL = "openai/gpt-oss-120b"

# HR answers are short, so a small ceiling keeps latency and cost down.
MAX_TOKENS = 2048

# Temperature 0 for consistent, factual answers rather than creative variation.
TEMPERATURE = 0

# gpt-oss is a reasoning model; "hidden" keeps its chain of thought out of the
# response so the returned content is the answer alone.
REASONING_FORMAT = "hidden"

# Used in the persona instruction so the assistant names the right employer.
COMPANY_NAME = "Meridian Softworks"

# Full legal name, shown in the frontend's branding area.
COMPANY_DISPLAY_NAME = "MERIDIAN SOFTWORKS PRIVATE LIMITED"

# The exact sentence the assistant must return when the context cannot answer
# the question. Defined once, as a single unbroken string, so the prompt and
# any test that asserts on the refusal compare against the same text.
NO_ANSWER_SENTENCE = (
    "I don't have that information in our HR policies - please contact HR directly."
)


# ---------------------------------------------------------------------------
# Raised when the app cannot run for a configuration reason: a missing API key
# or a missing vector store. The CLI prints it and exits; the Flask API turns
# it into a clear JSON error so the UI shows a configuration message rather
# than an unexplained stack trace.
# ---------------------------------------------------------------------------
class ConfigurationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Step 1: load API keys from .env into the process environment.
# GOOGLE_API_KEY serves the Gemini embeddings; GROQ_API_KEY serves the
# answer-generation model.
# ---------------------------------------------------------------------------
def load_api_key() -> None:
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        # Accept GEMINI_API_KEY as an alias, since both names are in circulation.
        if os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        else:
            raise ConfigurationError(
                "GOOGLE_API_KEY is not set. Add GOOGLE_API_KEY=your_key_here to "
                ".env (get a key at https://aistudio.google.com/apikey) and "
                "restart the backend."
            )


# ---------------------------------------------------------------------------
# Step 2: query-side embeddings. Same model as ingest.py, but task_type is
# deliberately RETRIEVAL_QUERY rather than RETRIEVAL_DOCUMENT - Gemini encodes
# a short question differently from a stored passage, and using the matching
# task type on each side is what makes the two comparable in vector space.
# ---------------------------------------------------------------------------
def build_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY",
    )


# ---------------------------------------------------------------------------
# Step 3: open the vector store that ingest.py persisted. Bail out with a
# plain-English message if the store is missing or empty, instead of letting
# Chroma raise a confusing stack trace deep inside a later search call.
# ---------------------------------------------------------------------------
def load_vector_store() -> Chroma:
    if not PERSIST_DIR.is_dir():
        raise ConfigurationError(
            f"No vector store found at ./{PERSIST_DIR}/. Build the knowledge "
            "base first:  venv/bin/python ingest.py"
        )

    load_api_key()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=build_query_embeddings(),
    )

    # A store can exist on disk yet hold no vectors - e.g. an ingest run that
    # was interrupted, or a collection name that does not match ingest.py.
    count = vector_store._collection.count()
    if count == 0:
        raise ConfigurationError(
            f"The '{COLLECTION_NAME}' collection in ./{PERSIST_DIR}/ is empty. "
            "Re-run the ingest step:  venv/bin/python ingest.py"
        )

    print(f"Loaded vector store: {count} chunks in collection '{COLLECTION_NAME}'")
    return vector_store


# ---------------------------------------------------------------------------
# Step 4: wrap the store in a retriever that returns the top 3 matches.
# ---------------------------------------------------------------------------
def build_retriever():
    vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


# Module-level singletons, built once and reused by the Flask routes added later.
retriever = None
llm = None


def get_retriever():
    """Lazily build the retriever so importing this module costs nothing."""
    global retriever
    if retriever is None:
        retriever = build_retriever()
    return retriever


# ---------------------------------------------------------------------------
# Step 5: the answer-generation model, served by Groq.
# temperature=0 gives consistent, factual answers instead of creative
# variation - important for policy questions, where the same question must
# always return the same entitlement.
# ---------------------------------------------------------------------------
def get_llm() -> ChatGroq:
    global llm
    if llm is None:
        load_dotenv()
        if not os.getenv("GROQ_API_KEY"):
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Add GROQ_API_KEY=your_key_here to .env "
                "(get a key at https://console.groq.com/keys) and restart the "
                "backend."
            )
        llm = ChatGroq(
            model=ANSWER_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_format=REASONING_FORMAT,
        )
    return llm


# ---------------------------------------------------------------------------
# Prompt part 1: the system instruction - persona plus grounding rules.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = f"""You are the HR Assistant for {COMPANY_NAME}.

Rules you must follow:
1. Identity: whenever you are asked to describe yourself, your role, your
   purpose, or how you behave, you identify yourself as the HR Assistant for
   {COMPANY_NAME}. You never describe yourself as a generic AI assistant, a
   language model, or a chatbot, and you never name the technology behind you.
2. Grounding: you answer strictly and only from the context provided below.
   You never use outside knowledge, general knowledge of employment law, or
   knowledge of other companies' policies, and you never make assumptions or
   inferences beyond what the context states.
3. Gaps: you never guess and you never fill a gap with plausible wording. When
   the context does not cover the question, rule 5 below governs your entire
   reply, including the identity statement required by rule 1.
4. Precision: you quote specific figures, day counts, and deadlines exactly as
   the context states them, and you do not round, convert, or recalculate them.
5. If the provided context does not contain enough information to answer the
   question, respond with exactly this sentence and nothing else:
   '{NO_ANSWER_SENTENCE}'
   Do not attempt to answer using outside knowledge in that case, and do not
   partially answer using unrelated context.
6. Formatting: you never write excerpt numbers, file names, or bracketed source
   markers into your answer. Sources are displayed separately, underneath your
   answer, by the application."""


# ---------------------------------------------------------------------------
# Prompt part 2: the retrieved chunks, joined and labelled as "Context:".
# ---------------------------------------------------------------------------
def format_context(retrieved_docs: list[Document]) -> str:
    if not retrieved_docs:
        return "Context:\n(No policy excerpts were retrieved for this question.)"

    blocks = []
    for position, doc in enumerate(retrieved_docs, start=1):
        # Label each excerpt with its origin so the answer can cite a source.
        meta = doc.metadata or {}
        source = Path(meta.get("source", "unknown")).name
        if meta.get("section"):
            locator = f"section '{meta['section']}'"
        elif meta.get("page_label") is not None:
            locator = f"page {meta['page_label']}"
        elif meta.get("page") is not None:
            locator = f"page {meta['page']}"
        else:
            locator = "location unknown"
        blocks.append(f"[Excerpt {position} - {source}, {locator}]\n{doc.page_content}")

    return "Context:\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Build the short citation line shown under an answer, e.g.
# "Leave Policy, Payments" or "Page 9, Page 10".
# A chunk's 'section' name is preferred; paginated sources fall back to their
# page number. Duplicates are dropped but first-appearance order is kept, so
# the line follows retrieval relevance rather than being re-sorted.
# ---------------------------------------------------------------------------
def get_citation(retrieved_docs: list[Document]) -> str:
    labels: list[str] = []

    for doc in retrieved_docs or []:
        meta = doc.metadata or {}

        if meta.get("section"):
            label = str(meta["section"])
        elif meta.get("page_label") is not None:
            # 'page_label' is the page number as printed in the document, which
            # is what a reader can actually turn to; 'page' is 0-indexed, so
            # using it directly would cite page 9 as "Page 8".
            label = f"Page {meta['page_label']}"
        elif meta.get("page") is not None:
            label = f"Page {meta['page']}"
        else:
            continue

        if label not in labels:  # de-duplicate, preserving first appearance
            labels.append(label)

    return ", ".join(labels)


# ---------------------------------------------------------------------------
# Prompt part 3: prior turns, as alternating "Human:" / "Assistant:" lines so
# Claude can resolve follow-up questions ("what about for interns?").
# Accepts (question, answer) pairs, {"role", "content"} dicts, or LangChain
# message objects, since the Flask layer may hand over any of these.
# ---------------------------------------------------------------------------
def format_chat_history(chat_history) -> str:
    if not chat_history:
        return ""

    lines: list[str] = []
    for turn in chat_history:
        # (question, answer) pair
        if isinstance(turn, (tuple, list)) and len(turn) == 2:
            human, assistant = turn
            lines.append(f"Human: {human}")
            if assistant:
                lines.append(f"Assistant: {assistant}")
            continue

        # {"role": "...", "content": "..."} dict
        if isinstance(turn, dict):
            role = str(turn.get("role", "")).lower()
            content = turn.get("content", "")
            speaker = "Human" if role in ("human", "user") else "Assistant"
            lines.append(f"{speaker}: {content}")
            continue

        # LangChain BaseMessage
        role = getattr(turn, "type", None) or getattr(turn, "role", "")
        content = getattr(turn, "content", str(turn))
        speaker = "Human" if str(role).lower() in ("human", "user") else "Assistant"
        lines.append(f"{speaker}: {content}")

    if not lines:
        return ""
    return "Conversation so far:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude may return content as a plain string or as a list of blocks (a
# thinking block followed by a text block, for instance). Pull out only the
# text so callers get a clean answer string, never raw block dicts.
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p).strip()

    return str(content).strip()


# ---------------------------------------------------------------------------
# Assemble all four parts into one prompt, send it to Claude as a single
# message, and return only the response text.
# ---------------------------------------------------------------------------
def generate_answer(question: str, retrieved_docs: list[Document], chat_history=None) -> str:
    parts = [
        SYSTEM_INSTRUCTION,            # 1. persona + rules
        format_context(retrieved_docs),  # 2. retrieved chunks
    ]

    history_block = format_chat_history(chat_history)  # 3. prior turns
    if history_block:
        parts.append(history_block)

    parts.append(f"Current question: {question}")  # 4. the question

    prompt = "\n\n".join(parts)

    response = get_llm().invoke([("human", prompt)])
    return extract_text(response.content)


# ---------------------------------------------------------------------------
# Despite rule 6, the model occasionally copies an "[Excerpt 2 - hr_policy.pdf,
# page 8]" label out of the context and into its answer. The citation line
# under the answer is the supported way to show sources, so strip those
# markers here as a safety net - raw metadata must never reach the UI.
# ---------------------------------------------------------------------------
SOURCE_MARKER_RE = re.compile(
    r"\s*[\[【(]\s*Excerpt\s*\d+[^\]】)]*[\]】)]", re.IGNORECASE
)


def strip_source_markers(answer: str) -> str:
    cleaned = SOURCE_MARKER_RE.sub("", answer)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)          # close " ." gaps
    cleaned = re.sub(r"([.!?])\s*\.(?=\s|$)", r"\1", cleaned)  # drop stray ".."
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Is this answer the exact fallback sentence? Compared leniently on curly
# quotes, dashes, whitespace and the trailing period, because a model can
# reproduce the sentence with typographic substitutions. Used to decide
# whether a citation is shown at all.
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip('"').rstrip(".").lower()


def is_fallback_answer(answer: str) -> bool:
    return _normalise(answer) == _normalise(NO_ANSWER_SENTENCE)


# ---------------------------------------------------------------------------
# Fail fast and clearly: check both API keys and the vector store up front,
# without making any network call. Used by the CLI below and by server.py.
# ---------------------------------------------------------------------------
def check_configuration() -> None:
    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise ConfigurationError(
            "GOOGLE_API_KEY is missing. Add it to .env and restart the backend."
        )
    if not os.getenv("GROQ_API_KEY"):
        raise ConfigurationError(
            "GROQ_API_KEY is missing. Add it to .env and restart the backend."
        )
    if not PERSIST_DIR.is_dir():
        raise ConfigurationError(
            f"No knowledge base found at ./{PERSIST_DIR}/. "
            "Run 'python ingest.py' to build it, then restart the backend."
        )


# ---------------------------------------------------------------------------
# Answer one question end to end: retrieve, generate, derive the citation,
# and decide whether the result is grounded. Both the CLI below and the
# /api/chat route in server.py call this, so the web UI and the terminal
# always behave identically.
# ---------------------------------------------------------------------------
def answer_question(question: str, chat_history=None) -> dict:
    docs = get_retriever().invoke(question)
    answer = strip_source_markers(generate_answer(question, docs, chat_history))
    citation = get_citation(docs)

    # No citation is ever shown for the fallback sentence.
    grounded = (not is_fallback_answer(answer)) and bool(citation)

    return {
        "answer": answer,
        "citation": citation if grounded else None,
        "grounded": grounded,
        "documents": docs,  # for the CLI's per-chunk listing only
    }


# ---------------------------------------------------------------------------
# Terminal smoke test, kept from before the UI existed:
#   venv/bin/python app.py --ask "how much maternity leave do I get?"
# ---------------------------------------------------------------------------
def run_cli(question: str) -> None:
    result = answer_question(question, chat_history=[])

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(result['documents'])} chunk(s):\n")
    for position, doc in enumerate(result["documents"], start=1):
        meta = doc.metadata
        locator = (
            f"section '{meta['section']}'"
            if meta.get("section")
            else f"page {meta.get('page_label', meta.get('page'))}"
        )
        print(f"  [{position}] {Path(meta['source']).name} - {locator}")

    print("\nAnswer:\n")
    print(result["answer"])
    # Mirror the UI rule: no citation under the fallback response.
    print(
        f"\nSource: {result['citation']}"
        if result["grounded"]
        else "\n(no source - fallback answer)"
    )
    print()


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in ("--ask", "-a"):
        args = args[1:]
    question = " ".join(args).strip()

    try:
        check_configuration()
        run_cli(question or "How much casual leave do I get?")
    except ConfigurationError as exc:
        sys.exit(f"\nCONFIGURATION ERROR: {exc}\n")


if __name__ == "__main__":
    main()
