"use client";

import { ArrowUp, LockKeyhole, Sparkles } from "lucide-react";
import { FormEvent, useState } from "react";
import { PageHeading } from "@/components/page-heading";
import { askAthena } from "@/lib/api";

const EXAMPLES = [
  "Who is most likely to win today?",
  "Which pitcher has the strongest strikeout outlook?",
  "Which predictions are too uncertain?",
  "Who is most likely to homer?",
];

export default function AskAthenaPage() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setWorking(true);
    setError(null);
    try {
      const response = await askAthena(question);
      setAnswer(response.answer);
    } catch {
      setError("Athena could not reach the prediction tools. Try again when the API is available.");
    } finally {
      setWorking(false);
    }
  }
  return (
    <div className="page ask-page">
      <PageHeading
        eyebrow="Grounded prediction agent"
        title="Ask the model, in plain language"
        description="Athena explains stored forecasts and their uncertainty. It never calculates a new probability in conversation."
      />
      <div className="agent-panel">
        <div className="agent-trust">
          <LockKeyhole size={16} />
          Answers are constrained to versioned prediction tools.
        </div>
        {answer ? (
          <div className="agent-answer">
            <span><Sparkles size={18} /></span>
            <p>{answer}</p>
          </div>
        ) : (
          <div className="agent-welcome">
            <div className="athena-orbit"><Sparkles size={26} /></div>
            <h2>What do you want to understand?</h2>
            <p>Ask about winners, run environments, pitcher strikeouts, batter outcomes, or why a forecast moved.</p>
          </div>
        )}
        {error && <p className="form-error" role="alert">{error}</p>}
        <div className="prompt-list">
          {EXAMPLES.map((example) => (
            <button key={example} onClick={() => setQuestion(example)}>{example}</button>
          ))}
        </div>
        <form className="ask-form" onSubmit={submit}>
          <label>
            <span className="sr-only">Question for Athena</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about today’s predictions…"
              rows={2}
            />
          </label>
          <button aria-label="Ask Athena" disabled={working || !question.trim()}>
            <ArrowUp size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
