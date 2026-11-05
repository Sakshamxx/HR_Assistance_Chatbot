import React, { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import Message from "./components/Message.jsx";
import ChatInput from "./components/ChatInput.jsx";

// ---------------------------------------------------------------------------
// Static content: topic shortcuts and the suggested follow-up questions.
// Each topic shortcut submits the question next to it.
// ---------------------------------------------------------------------------
const TOPICS = [
  { label: "Leave & Time Off", question: "What is the leave policy?" },
  { label: "Benefits", question: "What employee benefits are available?" },
  { label: "Workplace Policies", question: "What are the workplace conduct policies?" },
  { label: "Travel & Expenses", question: "What is the travel and expense policy?" },
  { label: "Onboarding", question: "What happens during onboarding and probation?" },
];

const FOLLOW_UPS = [
  "How do I apply for leave?",
  "Can I carry forward my leaves?",
  "What is the maternity leave policy?",
];

let messageId = 0;
const nextId = () => ++messageId;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [configError, setConfigError] = useState(null);
  const bottomRef = useRef(null);

  // -------------------------------------------------------------------------
  // On load, ask the backend whether its API keys and knowledge base are in
  // place, so a misconfiguration shows as a clear banner rather than as a
  // failure on the user's first question.
  // -------------------------------------------------------------------------
  useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => setConfigError(data.ok ? null : data.error))
      .catch(() =>
        setConfigError(
          "Cannot reach the HR Assistant backend. Start it with: python app.py",
        ),
      );
  }, []);

  // Keep the newest message in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  // -------------------------------------------------------------------------
  // Submit a question: show it immediately, show the loading state, then call
  // POST /api/chat with the question plus the conversation so far.
  // -------------------------------------------------------------------------
  const sendQuestion = useCallback(
    async (text) => {
      const question = text.trim();
      if (!question || loading) return;

      // History excludes error notices and the question being asked now.
      const history = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));

      setMessages((prev) => [...prev, { id: nextId(), role: "user", content: question }]);
      setLoading(true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, history }),
        });
        const data = await res.json();

        if (!res.ok) {
          if (data.configurationError) setConfigError(data.error);
          throw new Error(data.error || "The assistant is unavailable.");
        }

        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "assistant",
            content: data.answer,
            citation: data.citation,
            grounded: data.grounded,
          },
        ]);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "error", content: err.message },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, messages],
  );

  const isEmpty = messages.length === 0;
  const lastMessage = messages[messages.length - 1];
  const showFollowUps = !loading && lastMessage?.role === "assistant";
  const disabled = loading || Boolean(configError);

  return (
    <div className="app-shell">
      {/* Left: branding and topic navigation (hidden on small screens) */}
      <Sidebar topics={TOPICS} onPick={sendQuestion} disabled={disabled} />

      {/* Centre: the conversation */}
      <main className="chat-panel">
        <header className="chat-header">
          <div className="chat-header-brand">
            <span className="brand-mark">MS</span>
            <span className="brand-company">MERIDIAN SOFTWORKS PRIVATE LIMITED</span>
          </div>
          <h1>Your HR questions, answered instantly</h1>
          <p>
            Answers come directly from Meridian Softworks HR policy documents, with
            the source shown beneath each response.
          </p>
        </header>

        {/* Configuration problems surface here instead of as a stack trace */}
        {configError && (
          <div className="config-error" role="alert">
            <strong>Configuration needed</strong>
            <span>{configError}</span>
          </div>
        )}

        <div className="messages">
          {isEmpty && !configError && (
            <div className="empty-state">
              <div className="empty-avatar">HR</div>
              <p className="empty-title">Ask me about any company policy</p>
              <p className="empty-sub">
                Leave, benefits, conduct, travel, onboarding and more.
              </p>
              {/* Same shortcuts as the sidebar, for screens where it is hidden */}
              <div className="empty-topics">
                {TOPICS.map((topic) => (
                  <button
                    key={topic.label}
                    className="chip"
                    onClick={() => sendQuestion(topic.question)}
                    disabled={disabled}
                  >
                    {topic.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <Message key={message.id} message={message} />
          ))}

          {/* Animated loading state while retrieval and generation run */}
          {loading && (
            <div className="msg msg-assistant">
              <div className="avatar">HR</div>
              <div className="bubble bubble-assistant searching">
                <span>Searching company policies</span>
                <span className="dots">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
            </div>
          )}

          {/* Suggested follow-ups, shown under the latest answer */}
          {showFollowUps && (
            <div className="follow-ups">
              {FOLLOW_UPS.map((q) => (
                <button
                  key={q}
                  className="chip chip-followup"
                  onClick={() => sendQuestion(q)}
                  disabled={disabled}
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <ChatInput onSubmit={sendQuestion} disabled={disabled} loading={loading} />
      </main>

      {/* Right: secondary explainer panel (hidden on narrow screens) */}
      <HowItWorks />
    </div>
  );
}
