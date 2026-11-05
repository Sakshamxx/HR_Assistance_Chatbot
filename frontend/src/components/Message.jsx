import React from "react";

// ---------------------------------------------------------------------------
// Minimal rich-text rendering: the model replies with **bold** and "-" bullet
// lists, and showing those markers raw would look unpolished. This handles
// just those two cases, so no markdown dependency is needed.
// ---------------------------------------------------------------------------
function renderInline(text, keyPrefix) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      <React.Fragment key={`${keyPrefix}-${i}`}>{part}</React.Fragment>
    ),
  );
}

function RichText({ text }) {
  const blocks = [];
  let list = null;

  for (const raw of String(text).split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);

    if (bullet) {
      if (!list) list = [];
      list.push(bullet[1]);
      continue;
    }
    if (list) {
      blocks.push({ type: "ul", items: list });
      list = null;
    }
    if (line.trim()) blocks.push({ type: "p", text: line });
  }
  if (list) blocks.push({ type: "ul", items: list });

  return blocks.map((block, i) =>
    block.type === "ul" ? (
      <ul className="rt-list" key={i}>
        {block.items.map((item, j) => (
          <li key={j}>{renderInline(item, `${i}-${j}`)}</li>
        ))}
      </ul>
    ) : (
      <p className="rt-p" key={i}>
        {renderInline(block.text, i)}
      </p>
    ),
  );
}

// ---------------------------------------------------------------------------
// One chat message. User messages are indigo bubbles on the right; assistant
// messages are light cards with an avatar on the left.
//
// The citation renders ONLY when the backend reports grounded === true and a
// citation string exists - never a placeholder when there is no source.
// ---------------------------------------------------------------------------
export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble bubble-user">{message.content}</div>
      </div>
    );
  }

  const isError = message.role === "error";

  return (
    <div className="msg msg-assistant">
      <div className={`avatar${isError ? " avatar-error" : ""}`}>HR</div>
      <div className={`bubble bubble-assistant${isError ? " bubble-error" : ""}`}>
        <RichText text={message.content} />

        {message.grounded && message.citation && (
          <div className="citation">
            <span className="citation-glyph">▣</span>
            <span>Source · {message.citation}</span>
          </div>
        )}
      </div>
    </div>
  );
}
