import React, { useState } from "react";

// ---------------------------------------------------------------------------
// Bottom input: rounded field plus send button. Enter submits; the field is
// disabled while a question is in flight or while configuration is broken.
// ---------------------------------------------------------------------------
export default function ChatInput({ onSubmit, disabled, loading }) {
  const [value, setValue] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (!value.trim() || disabled) return;
    onSubmit(value);
    setValue("");
  };

  return (
    <form className="composer" onSubmit={submit}>
      <input
        className="composer-input"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask about company policies..."
        disabled={disabled}
        aria-label="Ask about company policies"
      />
      <button
        type="submit"
        className="send"
        disabled={disabled || !value.trim()}
        aria-label="Send question"
      >
        {loading ? (
          <span className="spinner" />
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M4 12h15M13 6l6 6-6 6"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </button>
    </form>
  );
}
