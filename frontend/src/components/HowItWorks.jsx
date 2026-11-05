import React from "react";

// ---------------------------------------------------------------------------
// Right column: a short explainer so employees understand where answers come
// from and why a source appears under some replies but not others.
// Hidden below 1200px.
// ---------------------------------------------------------------------------
const STEPS = [
  {
    title: "Policy search",
    body: "Your question is matched against the company's HR policy documents.",
  },
  {
    title: "Grounded answer",
    body: "The assistant answers using only those documents - never outside knowledge.",
  },
  {
    title: "Source shown",
    body: "Grounded answers display the policy section or page they came from.",
  },
];

export default function HowItWorks() {
  return (
    <aside className="how-it-works">
      <h2>How it works</h2>

      <ol className="steps">
        {STEPS.map((step, i) => (
          <li key={step.title}>
            <span className="step-num">{i + 1}</span>
            <div>
              <p className="step-title">{step.title}</p>
              <p className="step-body">{step.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <div className="note">
        <p className="note-title">
          <span className="citation-glyph">▣</span> About sources
        </p>
        <p>
          A source line appears beneath answers taken from policy documents. If a
          question is not covered, you will be directed to the HR team instead.
        </p>
      </div>
    </aside>
  );
}
