import { useState } from "react";
import { Link } from "react-router-dom";
import api from "../api/client.js";
import { useAuth } from "../context/AuthContext.jsx";

const METHOD_COLORS = {
  bm25:        { bg: "#fff8ed", border: "#e8d4a8", accent: "#9a6a16", label: "#7a5010" },
  word2vec:    { bg: "#f0f4ff", border: "#b8cbf0", accent: "#2558c4", label: "#1a3e8e" },
  transformer: { bg: "#edf5ec", border: "#b4d4b2", accent: "#2d7d2b", label: "#1e5c1c" },
};

const SAMPLE_QUERIES = [
  "What is the time limit for a PIO to respond to an RTI application?",
  "What are the penalties under IPC for cheating?",
  "How can a consumer file a complaint under the Consumer Protection Act?",
];

function ScoreBar({ score, max }) {
  const pct = max > 0 ? Math.min((score / max) * 100, 100) : 0;
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="score-bar-val">{score.toFixed(4)}</span>
    </div>
  );
}

function MethodColumn({ result, maxScore }) {
  const colors = METHOD_COLORS[result.method] || METHOD_COLORS.bm25;
  return (
    <div
      className="compare-col"
      style={{ "--col-bg": colors.bg, "--col-border": colors.border, "--col-accent": colors.accent }}
    >
      <div className="compare-col-head">
        <div className="compare-col-label" style={{ color: colors.label }}>
          {result.label}
        </div>
        <div className="compare-col-latency">{result.latency_ms} ms</div>
      </div>
      <p className="compare-col-desc">{result.description}</p>

      {result.hits.length === 0 ? (
        <div className="compare-empty-hits">No results</div>
      ) : (
        <ol className="compare-hits">
          {result.hits.map((hit, i) => (
            <li key={i} className="compare-hit">
              <div className="compare-hit-meta">
                <span className="compare-hit-rank">#{i + 1}</span>
                <span className="compare-hit-source" title={hit.source}>{hit.source}</span>
                {hit.section_number && (
                  <span className="compare-hit-section">§ {hit.section_number}</span>
                )}
                <ScoreBar score={hit.score} max={maxScore} />
              </div>
              {hit.section_title && (
                <div className="compare-hit-sec-title">{hit.section_title}</div>
              )}
              <p className="compare-hit-text">{hit.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export default function Compare() {
  const { user, logout } = useAuth();
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const initials = (user?.email || "?").slice(0, 1).toUpperCase();

  const runCompare = async (q) => {
    const text = (q ?? query).trim();
    if (!text || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const { data } = await api.post("/compare", { query: text, top_k: topK });
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Compare request failed.");
    } finally {
      setLoading(false);
    }
  };

  const maxScore = result
    ? Math.max(
        ...result.methods.flatMap((m) => m.hits.map((h) => h.score)),
        0.0001,
      )
    : 1;

  return (
    <div className="compare-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-head">
          <div className="brand">
            <span className="brand-mark">⚖</span>
            <span className="brand-name">NyayaBot</span>
          </div>
          <Link to="/chat" className="new-btn" style={{ textDecoration: "none", textAlign: "center" }}>
            ← Back to chat
          </Link>
        </div>
        <div className="compare-sidebar-info">
          <div className="compare-sidebar-title">NLP Method Comparison</div>
          <p className="compare-sidebar-body">
            Run the same legal query through three retrieval approaches side by side and compare their results.
          </p>
          <div className="compare-method-legend">
            {[
              { key: "bm25",        label: "BM25",        note: "Lexical" },
              { key: "word2vec",    label: "Word2Vec",    note: "RNN-era" },
              { key: "transformer", label: "MiniLM",      note: "Transformer" },
            ].map((m) => (
              <div key={m.key} className="compare-legend-row">
                <span
                  className="compare-legend-dot"
                  style={{ background: METHOD_COLORS[m.key].accent }}
                />
                <span className="compare-legend-label">{m.label}</span>
                <span className="compare-legend-note muted small">{m.note}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="sidebar-foot">
          <div className="user-row">
            <div className="avatar">{initials}</div>
            <div className="user-meta">
              <div className="user-email" title={user?.email}>{user?.email}</div>
              <button className="link" onClick={logout}>Sign out</button>
            </div>
          </div>
        </div>
      </aside>

      {/* Main area */}
      <main className="compare-main">
        <header className="chat-header">
          <div className="chat-header-left">
            <div className="chat-title">NLP Retrieval Comparison</div>
            <div className="chat-sub muted small">
              BM25 (Classic) · Word2Vec (RNN-era) · MiniLM Transformer (Semantic)
            </div>
          </div>
        </header>

        {/* Query form */}
        <div className="compare-form-wrap">
          <div className="compare-samples">
            <span className="muted small">Try: </span>
            {SAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                className="compare-sample-btn"
                onClick={() => { setQuery(q); runCompare(q); }}
                disabled={loading}
              >
                {q}
              </button>
            ))}
          </div>
          <form
            className="compare-form"
            onSubmit={(e) => { e.preventDefault(); runCompare(); }}
          >
            <textarea
              className="compare-textarea"
              rows={2}
              placeholder="Enter a legal query to compare all three retrieval methods…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
            />
            <div className="compare-form-controls">
              <label className="compare-topk-label">
                Top-k results
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  disabled={loading}
                  className="compare-topk-select"
                >
                  {[3, 5, 8, 10].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
              <button
                type="submit"
                className="compare-run-btn"
                disabled={loading || !query.trim()}
              >
                {loading ? "Running…" : "Compare methods"}
              </button>
            </div>
          </form>
          {error && <div className="upload-error small">{error}</div>}
        </div>

        {/* Results */}
        {result && (
          <div className="compare-results">
            {/* NLP Analysis header */}
            <div className="compare-analysis-bar">
              <div className="compare-analysis-item">
                <span className="compare-analysis-label">Detected intent</span>
                <span className="badge-intent">{result.intent_label}</span>
                <span className="muted small">({Math.round(result.intent_confidence * 100)}% confidence)</span>
              </div>
              <div className="compare-analysis-item">
                <span className="compare-analysis-label">Normalized query</span>
                <code className="compare-normalized">{result.normalized_query}</code>
              </div>
            </div>

            {/* Three columns */}
            <div className="compare-cols">
              {result.methods.map((m) => (
                <MethodColumn key={m.method} result={m} maxScore={maxScore} />
              ))}
            </div>
          </div>
        )}

        {!result && !loading && (
          <div className="compare-placeholder">
            <div className="empty-mark">⚖</div>
            <p className="muted">
              Enter a query above to see how BM25, Word2Vec, and MiniLM Transformer each rank the legal corpus.
            </p>
          </div>
        )}

        {loading && (
          <div className="compare-placeholder">
            <div className="typing" style={{ justifyContent: "center" }}>
              <span /><span /><span />
            </div>
            <p className="muted small">Running all three retrieval methods…</p>
          </div>
        )}
      </main>
    </div>
  );
}
