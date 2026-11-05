import React from "react";

// ---------------------------------------------------------------------------
// Left column: company branding, the product name, and the topic shortcuts.
// Hidden below 900px so the conversation stays the focus on small screens.
// ---------------------------------------------------------------------------
export default function Sidebar({ topics, onPick, disabled }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">MS</span>
        <div className="brand-text">
          <span className="brand-company">MERIDIAN SOFTWORKS</span>
          <span className="brand-company-sub">PRIVATE LIMITED</span>
        </div>
      </div>

      <div className="product">
        <span className="product-name">HR Assistant</span>
        <span className="product-tag">Internal policy support</span>
      </div>

      <nav className="topics">
        <p className="topics-label">Browse topics</p>
        {topics.map((topic) => (
          <button
            key={topic.label}
            className="topic"
            onClick={() => onPick(topic.question)}
            disabled={disabled}
          >
            {topic.label}
          </button>
        ))}
      </nav>

      <p className="sidebar-foot">Policy data current as of the latest handbook release.</p>
    </aside>
  );
}
