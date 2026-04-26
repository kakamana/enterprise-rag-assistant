"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type Source = {
  doc_id: string;
  domain: string;
  paragraph_id: number;
  text: string;
  score: number;
};

type AskResponse = {
  answer: string;
  sources: Source[];
  faithfulness_score: number;
  retrieval_recall: number;
  abstained: boolean;
  decision_aid_disclaimer: string;
};

const DEMO_QUESTIONS = [
  "What is the limit for expense reimbursement?",
  "Who approves a vendor onboarding?",
  "How quickly must I report a near-miss?",
  "What is the policy on remote work?",
];

export default function Home() {
  const [question, setQuestion] = useState(DEMO_QUESTIONS[0]);
  const [resp, setResp] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function ask() {
    setLoading(true);
    try {
      const r = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
      });
      setResp(await r.json());
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">Enterprise RAG Assistant</h1>
      <p className="opacity-70 mb-6">
        Ask a policy question. Get a grounded answer, the cited paragraphs, and a faithfulness score.
      </p>

      <div className="rounded-2xl border bg-white p-4 mb-4">
        <textarea
          className="w-full border rounded-xl p-3 text-base"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div className="mt-2 flex flex-wrap gap-2">
          {DEMO_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => setQuestion(q)}
              className="text-xs rounded-full border px-3 py-1 hover:bg-neutral-100"
            >
              {q}
            </button>
          ))}
        </div>
        <button
          onClick={ask}
          disabled={loading}
          className="mt-3 rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
        >
          {loading ? "Retrieving..." : "Ask"}
        </button>
      </div>

      {resp && (
        <>
          <div className="rounded-2xl border bg-white p-4 mb-4">
            <div className="text-xs uppercase opacity-60 mb-1">Answer</div>
            <p className="text-base">{resp.answer}</p>
            {resp.abstained && (
              <p className="text-xs text-amber-700 mt-2 italic">
                Model abstained because faithfulness fell below the 0.50 gate.
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <FaithGauge value={resp.faithfulness_score} />
            <Stat
              label="Retrieval recall (proxy)"
              value={(resp.retrieval_recall * 100).toFixed(0) + "%"}
            />
          </div>

          <div className="rounded-2xl border bg-white p-4">
            <div className="text-xs uppercase opacity-60 mb-2">Cited sources</div>
            <ul className="space-y-3">
              {resp.sources.map((s, i) => (
                <li key={i} className="border-l-4 border-neutral-300 pl-3">
                  <div className="text-xs opacity-60">
                    {s.domain} · {s.doc_id} · p{s.paragraph_id} · score {s.score.toFixed(2)}
                  </div>
                  <div className="text-sm">{s.text}</div>
                </li>
              ))}
            </ul>
          </div>

          <p className="mt-6 text-xs opacity-60 italic">
            {resp.decision_aid_disclaimer}
          </p>
        </>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border bg-white p-4">
      <div className="text-xs uppercase tracking-wide opacity-60">{label}</div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
    </div>
  );
}

function FaithGauge({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  const band =
    pct >= 0.8 ? "bg-emerald-500" : pct >= 0.5 ? "bg-amber-500" : "bg-rose-500";
  return (
    <div className="rounded-2xl border bg-white p-4">
      <div className="text-xs uppercase tracking-wide opacity-60">Faithfulness</div>
      <div className="text-2xl font-semibold mt-1">{(pct * 100).toFixed(0)}%</div>
      <div className="mt-2 h-2 w-full bg-neutral-200 rounded-full overflow-hidden">
        <div className={`h-2 ${band}`} style={{ width: `${pct * 100}%` }} />
      </div>
    </div>
  );
}
