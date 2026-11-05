#!/usr/bin/env python3
"""
server.py - Flask API for the HR Assistant.

A thin HTTP layer over the RAG pipeline in app.py. It contains no retrieval
or generation logic of its own: it calls answer_question(), which runs the
same retriever, generate_answer() and get_citation() the CLI uses.

Start the backend:   venv/bin/python server.py
"""

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Everything below comes from the existing RAG module - nothing is redefined.
from app import (
    COMPANY_DISPLAY_NAME,
    ConfigurationError,
    answer_question,
    check_configuration,
    get_retriever,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# The React dev server runs on a different port, so it needs CORS in development.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Cap how much conversation history goes into the prompt on long chats.
MAX_HISTORY_MESSAGES = 10

# macOS uses port 5000 for AirPlay Receiver, so default to 5001 instead.
BACKEND_PORT = int(os.getenv("PORT", "5001"))


# ---------------------------------------------------------------------------
# GET /api/health - lets the UI show a configuration banner on load instead
# of waiting for the user's first question to fail.
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    try:
        check_configuration()
    except ConfigurationError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    return jsonify({"ok": True, "company": COMPANY_DISPLAY_NAME})


# ---------------------------------------------------------------------------
# POST /api/chat - the one endpoint the frontend needs.
#   in : {"question": "...", "history": [{"role": "user", "content": "..."}]}
#   out: {"answer": "...", "citation": "Page 9", "grounded": true}
#        {"answer": "<fallback>", "citation": null, "grounded": false}
# 'grounded' is false whenever the answer is the fallback sentence or no
# source could be derived, so the UI knows never to render a citation.
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    history = payload.get("history")
    history = history[-MAX_HISTORY_MESSAGES:] if isinstance(history, list) else []

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        check_configuration()
        result = answer_question(question, history)  # retrieve -> generate -> cite
    except ConfigurationError as exc:
        return jsonify({"error": str(exc), "configurationError": True}), 503
    except Exception as exc:  # upstream API down, network error, bad key
        app.logger.exception("chat request failed")
        return (
            jsonify(
                {
                    "error": "The assistant could not reach its language model. "
                    f"Please try again. ({type(exc).__name__})"
                }
            ),
            502,
        )

    # 'documents' holds raw chunks for the CLI listing - never send it to the UI.
    return jsonify(
        {
            "answer": result["answer"],
            "citation": result["citation"],
            "grounded": result["grounded"],
        }
    )


# ---------------------------------------------------------------------------
# Entry point: verify configuration, warm the vector store so the first
# question is not slow, then serve.
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        check_configuration()
        get_retriever()
    except ConfigurationError as exc:
        sys.exit(f"\nCONFIGURATION ERROR: {exc}\n")

    print(f"\n  {COMPANY_DISPLAY_NAME} - HR Assistant backend")
    print(f"  API ready at http://127.0.0.1:{BACKEND_PORT}/api/chat")
    print("  Start the frontend with:  cd frontend && npm run dev\n")
    app.run(host="127.0.0.1", port=BACKEND_PORT, debug=False)


if __name__ == "__main__":
    main()
