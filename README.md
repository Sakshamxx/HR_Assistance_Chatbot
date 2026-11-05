<div align="center">

# 🏢 HR Assistant

### A production-shaped RAG chatbot that answers HR policy questions — and admits when it can't.

Employees ask questions in plain language. Answers come **only** from the
company's own policy documents, each with the page or section it came from. When
the documents don't cover a question, the assistant says so instead of
inventing an answer.

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.4-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-1.5-FF6F61?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

</div>

<!-- ───────────────────────────────────────────────────────────────────────
     DEMO: add a screenshot or short GIF at docs/screenshot.png, then
     uncomment the block below. A visual at the top is the single highest
     impact addition to this README.

<div align="center">
  <img src="docs/screenshot.png" width="900" alt="HR Assistant chat interface" />
</div>
     ─────────────────────────────────────────────────────────────────── -->

---

## 🎯 The problem this solves

General-purpose chatbots answer HR questions confidently and often wrongly —
they blend real policy with training-data guesses about employment law. For HR,
a plausible-sounding wrong answer about notice periods or leave entitlement is
worse than no answer at all.

This project constrains the model to a single source of truth, and makes
**refusal a first-class behaviour**:

| Question | Behaviour |
|---|---|
| "How many leave days do I get per year?" | Answers from the handbook → `▣ Source · Page 8, Page 9` |
| "What is the maternity leave policy?" | Handbook defers to statute without a figure → **refuses**, no source shown |
| "What is the capital of France?" | Out of scope → **refuses** with the same fixed sentence |

---

## 🔬 What this project demonstrates

- **RAG beyond the tutorial** — hybrid chunking that keeps Markdown section
  names *or* PDF page numbers as metadata, so citations survive a second
  size-based split
- **Asymmetric embeddings** — `RETRIEVAL_DOCUMENT` at index time,
  `RETRIEVAL_QUERY` at search time, the way Gemini's embedding model is designed
  to be used
- **Grounding you can verify** — a `grounded` boolean computed server-side, so
  the UI can never render a citation under an ungrounded answer
- **Prompt engineering with a testable contract** — one exact fallback sentence,
  asserted against in tests rather than eyeballed
- **Clean separation** — `app.py` (pipeline) has no web code; `server.py` (API)
  has no retrieval logic; the React layer has no business rules
- **Honest engineering** — see [Known limitations](#-known-limitations) below

---

## 🧭 Architecture

```
          ┌──────────────────────────────────────────────────────────┐
          │                      ingest.py                           │
          │  knowledge_base/*.pdf .docx .md .txt                     │
          │       │                                                  │
          │       ├── type-aware loaders (PyPDF / Docx2txt / Text)   │
          │       ├── hybrid split: Markdown headers  OR  PDF pages  │
          │       ├── size cap: 700 chars, 100 overlap               │
          │       └── Gemini embeddings  (RETRIEVAL_DOCUMENT)        │
          │                      ↓                                   │
          │              ./chroma_db  (persistent, 3072-dim)         │
          └──────────────────────────────────────────────────────────┘
                                 ↓
   ┌───────────────┐      ┌──────────────┐      ┌────────────────────┐
   │  React UI     │ ───▶ │  server.py   │ ───▶ │      app.py        │
   │  :5173        │      │  Flask API   │      │   RAG pipeline     │
   │               │      │  /api/chat   │      │                    │
   │  question +   │      │              │      │  1. retrieve k=3   │
   │  history      │ ◀─── │  answer +    │ ◀─── │  2. generate       │
   │               │      │  citation +  │      │  3. cite + ground  │
   └───────────────┘      │  grounded    │      └────────────────────┘
                          └──────────────┘
```

**Why two embedding task types?** `ingest.py` embeds passages with
`task_type="RETRIEVAL_DOCUMENT"`; `app.py` embeds the question with
`task_type="RETRIEVAL_QUERY"`. Gemini encodes a short question differently from
a stored passage, and matching the task type on each side is what makes the two
comparable in vector space. The **model name must stay identical** across both
files — a mismatch fails silently, returning plausible-looking noise.

---

## 🛠 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `gemini-embedding-001` | Current GA model, 3072 dimensions |
| Generation | `openai/gpt-oss-120b` via Groq | Fast inference; `temperature=0` for repeatability |
| Vector store | Chroma (persistent) | Zero-config local persistence |
| Orchestration | LangChain 1.x | Loaders, splitters, retriever abstraction |
| Backend | Flask + flask-cors | One endpoint — nothing heavier is justified |
| Frontend | React 19 + Vite | 4 dependencies, hand-written CSS, no UI framework |

---

## 🚀 Quick start

### 1 · Clone and set up Python

```bash
git clone https://github.com/Sakshamxx/HR_Assistance_Chatbot.git
cd HR_Assistance_Chatbot

python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2 · Add API keys

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_google_api_key_here      # https://aistudio.google.com/apikey
GROQ_API_KEY=your_groq_api_key_here          # https://console.groq.com/keys
```

### 3 · Add policy documents and build the index

Drop any `.pdf`, `.docx`, `.md` or `.txt` policy files into `knowledge_base/`,
then:

```bash
python ingest.py
```

```
Reading documents from knowledge_base/ ...
  loaded   knowledge_base/hr_policy.pdf  (15 page/doc object(s))

Splitting documents ...
  split    knowledge_base/hr_policy.pdf  -> 15 page chunk(s), page numbers kept

  15 structural chunk(s) -> 57 final chunk(s) (max 700 chars, 100 overlap)

Embedding 57 chunk(s) with gemini-embedding-001 ...

====================================================================
INGEST COMPLETE
====================================================================
  Source documents loaded : 1
  Chunks created          : 57
  Vectors stored in Chroma: 57
```

### 4 · Run

```bash
# terminal 1 — backend
python server.py                  # http://127.0.0.1:5001

# terminal 2 — frontend
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**

> **Note** · The backend uses port **5001**, not 5000 — macOS reserves 5000 for
> AirPlay Receiver. Override with `PORT=5050 python server.py`.

---

## 💻 Terminal mode

The pipeline runs without the frontend:

```bash
python app.py --ask "how many leave days do I get per year?"
```

```
Question: how many leave days do I get per year?
Retrieved 3 chunk(s):

  [1] hr_policy.pdf - page 8
  [2] hr_policy.pdf - page 9
  [3] hr_policy.pdf - page 8

Answer:

Full-time employees, including probationers, are entitled to 18 days of leave
per year, accruing at 1.5 days each month.

Source: Page 8, Page 9
```

---

## 🔌 API

### `POST /api/chat`

```json
{
  "question": "how many leave days do I get per year?",
  "history": [
    { "role": "user", "content": "what is the notice period?" },
    { "role": "assistant", "content": "Sixty (60) calendar days." }
  ]
}
```

**Grounded** — the UI renders the citation:

```json
{
  "answer": "Full-time employees are entitled to 18 days of leave per year...",
  "citation": "Page 8, Page 9",
  "grounded": true
}
```

**Not covered by policy** — `grounded: false`, so **no source element renders**:

```json
{
  "answer": "I don't have that information in our HR policies - please contact HR directly.",
  "citation": null,
  "grounded": false
}
```

### `GET /api/health`

Reports whether both API keys and the vector store are present, so the UI shows
a configuration banner on load instead of failing on the first question.

| Status | Response |
|---|---|
| `200` | `{"ok": true, "company": "..."}` |
| `503` | `{"ok": false, "error": "GROQ_API_KEY is missing. Add it to .env..."}` |

---

## 📁 Project structure

```
.
├── ingest.py                 # builds the vector knowledge base
├── app.py                    # RAG pipeline — retriever, prompt, citations
├── server.py                 # Flask API (imports app.py, adds no logic)
├── requirements.txt
├── .env.example
├── knowledge_base/           # your policy documents  (git-ignored)
├── chroma_db/                # generated vector store (git-ignored)
└── frontend/
    ├── vite.config.js        # proxies /api → Flask :5001
    └── src/
        ├── App.jsx           # layout, state, API calls
        ├── styles.css        # design tokens + responsive layout
        └── components/
            ├── Message.jsx       # bubbles, avatar, citation pill
            ├── Sidebar.jsx       # branding + topic shortcuts
            ├── HowItWorks.jsx    # explainer panel
            └── ChatInput.jsx     # composer
```

---

## 🧠 Engineering decisions

<details>
<summary><b>Hybrid chunking: headers where they exist, pages otherwise</b></summary>

<br/>

Markdown documents split on `##` headers so each chunk carries its section name
(`section` metadata). PDFs without headers split per page instead, keeping the
page number (`page_label`). Both then pass through a 700-character recursive
split that **preserves the metadata from the first pass** — so a long section
becomes several chunks that all still cite the same source. Verified: a 3.6 KB
Markdown section produced 4 chunks, every one retaining its section name.

</details>

<details>
<summary><b>Citations use the printed page number, not the array index</b></summary>

<br/>

PyPDFLoader attaches a 0-indexed `page` plus `page_label`, the number printed on
the page. `get_citation()` prefers `page_label` — citing page 9 as "Page 8" is
wrong to anyone who opens the document to check. Duplicates collapse while
keeping first-appearance order, so three chunks from one page read `Page 9`
rather than `Page 9, Page 9, Page 9`.

</details>

<details>
<summary><b>One exact sentence for uncovered questions</b></summary>

<br/>

The system prompt mandates a fixed response when the context is insufficient:

> I don't have that information in our HR policies - please contact HR directly.

It lives in one constant, `NO_ANSWER_SENTENCE`, interpolated into the prompt and
compared leniently (curly quotes, en-dashes, whitespace, trailing period) — a
model will happily reproduce the sentence with typographic substitutions, and a
strict `==` would then mark a refusal as grounded and attach a bogus citation.

</details>

<details>
<summary><b>Grounding is decided on the server, not in the UI</b></summary>

<br/>

`grounded` is `true` only when the answer is not the fallback sentence **and** a
citation could be derived. React renders the source element solely on that flag,
so the interface can never show "Source: None" or an empty citation pill.

</details>

<details>
<summary><b>temperature = 0</b></summary>

<br/>

A policy question must return the same entitlement every time it is asked.
Creative variation is a defect here, not a feature.

</details>

---

## 🔭 Known limitations

Deliberately listed rather than hidden — each is a real, reproducible gap:

- **Follow-ups retrieve on literal text.** Conversation history reaches the LLM
  but not the retriever, so "is that in calendar days or working days?" searches
  for those words rather than the resolved question. Fix: condense the
  follow-up into a standalone query before retrieval.
- **PDF whitespace is not normalised.** pypdf emits a newline between words in
  some PDFs, costing ~23% of each chunk's capacity (81 useful words per chunk
  instead of 105).
- **Page citations where sections would be better.** PDF headings are visual
  (font size), not Markdown `##`, so the section-aware path never activates for
  PDF sources.
- **No evaluation harness.** Grounding and refusal are verified by scripted
  manual checks, not a scored regression suite.

---

## 📝 Note on the sample data

"MERIDIAN SOFTWORKS PRIVATE LIMITED" is a fictional company used for this
demo. The HR handbook used during development is **not** included in this
repository — add your own policy documents to `knowledge_base/` as described
above.

---

<div align="center">

**Saksham Chauhan** · [github.com/Sakshamxx](https://github.com/Sakshamxx)

Released under the [MIT License](LICENSE).

</div>
