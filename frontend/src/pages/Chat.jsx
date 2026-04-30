import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import api from "../api/client.js";
import { useAuth } from "../context/AuthContext.jsx";

const SUGGESTED = [
  "What is the time limit for a Public Information Officer to respond to an RTI application?",
  "What information is exempt from disclosure under the RTI Act?",
  "What is the fee for filing an RTI application?",
  "Who can file an appeal under the RTI Act?",
];

const SCENARIO_PREFIX = "[SCENARIO]";

const WHO_OPTIONS = ["landlord", "employer", "police", "government", "seller", "bank", "builder", "hospital", "other"];

const WHERE_OPTIONS = [
  "Pan-India", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
  "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
  "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
  "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
  "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
  "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi",
  "Jammu & Kashmir", "Ladakh",
];

const OUTCOME_OPTIONS = [
  { value: "know my rights", label: "Know my rights" },
  { value: "file a complaint", label: "File a complaint" },
  { value: "get compensation", label: "Get compensation" },
  { value: "understand procedure", label: "Understand procedure" },
  { value: "other", label: "Other" },
];

function isScenarioMessage(content) {
  return typeof content === "string" && content.startsWith(SCENARIO_PREFIX);
}

function parseScenarioMessage(content) {
  try {
    const jsonPart = content.slice(SCENARIO_PREFIX.length).split("\n")[0];
    return JSON.parse(jsonPart);
  } catch {
    return null;
  }
}

function ScenarioCard({ data }) {
  if (!data) return <span className="muted">[Scenario message]</span>;
  return (
    <div className="scenario-card">
      <div className="scenario-card-title">Legal Scenario</div>
      <dl className="scenario-fields">
        {data.what && (<><dt>What happened</dt><dd>{data.what}</dd></>)}
        {data.who && (<><dt>Who is involved</dt><dd className="capitalize">{data.who}</dd></>)}
        {data.when && (<><dt>When</dt><dd>{data.when}</dd></>)}
        {data.where && (<><dt>Where</dt><dd>{data.where}</dd></>)}
        {data.outcome && (<><dt>What I want</dt><dd className="capitalize">{data.outcome}</dd></>)}
      </dl>
    </div>
  );
}

function buildScenarioQuery(sc) {
  const outcomeVerb = sc.outcome === "other" ? "understand my options" : sc.outcome;
  const natural =
    `Situation: My ${sc.who} ${sc.what.trim()} in ${sc.where}${sc.when ? " " + sc.when.trim() : ""}. ` +
    `I want to ${outcomeVerb}. What are my legal rights and remedies?`;
  const structured = JSON.stringify({
    what: sc.what,
    who: sc.who,
    when: sc.when,
    where: sc.where,
    outcome: sc.outcome,
  });
  return `${SCENARIO_PREFIX}${structured}\n${natural}`;
}

export default function Chat() {
  const { user, logout } = useAuth();
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("nyayabot-theme") === "dark";
  });
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const [sessionDoc, setSessionDoc] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const [scenarioMode, setScenarioMode] = useState(false);
  const [scenario, setScenario] = useState({
    what: "",
    who: "landlord",
    when: "",
    where: "Pan-India",
    outcome: "know my rights",
  });

  useEffect(() => {
    const theme = darkMode ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("nyayabot-theme", theme);
  }, [darkMode]);

  const activeSession = useMemo(
    () => sessions.find((s) => s.id === activeId) || null,
    [sessions, activeId],
  );

  const loadSessions = async () => {
    const { data } = await api.get("/sessions");
    setSessions(data);
    return data;
  };

  useEffect(() => {
    loadSessions().then((list) => {
      if (list.length > 0) selectSession(list[0].id);
    });
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, sending]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [input]);

  const selectSession = async (id) => {
    setActiveId(id);
    setSessionDoc(null);
    setUploadError("");
    const [{ data: msgs }, docRes] = await Promise.all([
      api.get(`/sessions/${id}/messages`),
      api.get(`/documents/${id}`).catch(() => ({ data: null })),
    ]);
    setMessages(msgs);
    setSessionDoc(docRes.data || null);
  };

  const newSession = async () => {
    const { data } = await api.post("/sessions", {});
    setSessions((s) => [data, ...s]);
    setActiveId(data.id);
    setMessages([]);
    setSessionDoc(null);
    setUploadError("");
  };

  const sendText = async (text) => {
    if (!text || sending) return;

    let sessionId = activeId;
    if (!sessionId) {
      const { data } = await api.post("/sessions", {});
      setSessions((s) => [data, ...s]);
      sessionId = data.id;
      setActiveId(sessionId);
    }

    const optimistic = {
      id: `tmp-${Date.now()}`,
      session_id: sessionId,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((m) => [...m, optimistic]);
    setInput("");
    setSending(true);

    try {
      const { data } = await api.post("/chat", { session_id: sessionId, message: text });
      setMessages((m) => [
        ...m,
        {
          id: `a-${Date.now()}`,
          session_id: sessionId,
          role: "assistant",
          content: data.answer,
          created_at: new Date().toISOString(),
          refused: data.refused,
          sources: data.sources || [],
          follow_ups: data.follow_ups || [],
          intent_domain: data.intent_domain || null,
          intent_label: data.intent_label || null,
          top_span: data.top_span || null,
        },
      ]);
      loadSessions();
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          id: `err-${Date.now()}`,
          session_id: sessionId,
          role: "assistant",
          content: e?.response?.data?.detail || "Request failed.",
          created_at: new Date().toISOString(),
          error: true,
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const sendScenario = async () => {
    if (!scenario.what.trim() || sending) return;
    const message = buildScenarioQuery(scenario);
    setScenarioMode(false);
    setScenario({ what: "", who: "landlord", when: "", where: "Pan-India", outcome: "know my rights" });
    await sendText(message);
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (!file) return;

    if (file.type !== "application/pdf") {
      setUploadError("Only PDF files are accepted.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setUploadError("File exceeds the 10 MB limit.");
      return;
    }

    let sessionId = activeId;
    if (!sessionId) {
      const { data } = await api.post("/sessions", {});
      setSessions((s) => [data, ...s]);
      sessionId = data.id;
      setActiveId(sessionId);
    }

    setUploading(true);
    setUploadError("");
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("file", file);
      const { data } = await api.post("/documents/upload", form);
      setSessionDoc(data);
      loadSessions();
    } catch (err) {
      setUploadError(err?.response?.data?.detail || "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const onSubmit = (e) => {
    e.preventDefault();
    sendText(input.trim());
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendText(input.trim());
    }
  };

  const initials = (user?.email || "?").slice(0, 1).toUpperCase();

  return (
    <div className="chat-shell">
      <aside className="sidebar">
        <div className="sidebar-head">
          <div className="brand">
            <span className="brand-mark">⚖</span>
            <span className="brand-name">NyayaBot</span>
          </div>
          <button className="new-btn" onClick={newSession}>
            <span>+</span> New chat
          </button>
        </div>
        <ul className="sessions">
          {sessions.map((s) => (
            <li
              key={s.id}
              className={s.id === activeId ? "active" : ""}
              onClick={() => selectSession(s.id)}
              title={s.title}
            >
              {s.doc_id && <span className="session-doc-dot" title="Has document" />}
              {s.title}
            </li>
          ))}
          {sessions.length === 0 && <li className="muted small">No chats yet</li>}
        </ul>
        <div className="sidebar-foot">
          <Link to="/compare" className="compare-nav-link">
            Compare NLP Methods
          </Link>
          <div className="user-row">
            <div className="avatar">{initials}</div>
            <div className="user-meta">
              <div className="user-email" title={user?.email}>{user?.email}</div>
              <button className="link" onClick={logout}>Sign out</button>
            </div>
          </div>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-left">
            <div className="chat-title">{activeSession?.title || "New conversation"}</div>
            <div className="chat-sub muted small">Grounded in your ingested legal documents</div>
            {sessionDoc && (
              <div className="doc-indicator">
                <span className="doc-indicator-icon">📄</span>
                <span className="doc-indicator-name" title={sessionDoc.display_name}>
                  {sessionDoc.display_name}
                </span>
                <span className="doc-indicator-badge">{sessionDoc.chunk_count} chunks</span>
              </div>
            )}
          </div>
          <button
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
            title="Toggle dark mode"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </header>

        <div className="messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="empty">
              <div className="empty-mark">⚖</div>
              <h2>Ask about the ingested legal documents</h2>
              <p className="muted">Try one of these to get started:</p>
              <div className="suggestions">
                {SUGGESTED.map((q) => (
                  <button key={q} className="suggestion" onClick={() => sendText(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m) => (
            <div
              key={m.id}
              className={`msg msg-${m.role}${m.refused ? " refused" : ""}${m.error ? " error-msg" : ""}`}
            >
              <div className="msg-avatar" aria-hidden>
                {m.role === "user" ? initials : "⚖"}
              </div>
              <div className="msg-body">
                <div className="msg-meta">
                  <span className="msg-role">{m.role === "user" ? "You" : "NyayaBot"}</span>
                  {m.refused && <span className="badge badge-warn">out of scope</span>}
                  {m.error && <span className="badge badge-err">error</span>}
                  {m.role === "assistant" && m.intent_label && (
                    <span className="badge badge-intent" title={`Domain: ${m.intent_domain}`}>
                      {m.intent_label}
                    </span>
                  )}
                </div>
                <div className="msg-content">
                  {m.role === "assistant" ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  ) : isScenarioMessage(m.content) ? (
                    <ScenarioCard data={parseScenarioMessage(m.content)} />
                  ) : (
                    m.content
                  )}
                </div>
                {m.role === "assistant" && m.top_span && (
                  <div className="top-span-box">
                    <span className="top-span-label">Key clause</span>
                    <blockquote className="top-span-text">{m.top_span}</blockquote>
                  </div>
                )}
                {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                  <div className="sources">
                    <span className="sources-label">Sources</span>
                    {m.sources.map((s, i) => (
                      <span
                        key={`${s.source}-${i}`}
                        className={`source-chip${s.origin === "user_doc" ? " source-chip--user" : ""}`}
                        title={`similarity ${s.score.toFixed(3)}`}
                      >
                        <span className="dot" />
                        {s.origin === "user_doc" && (
                          <span className="source-origin-label">Your Doc · </span>
                        )}
                        {s.source}
                        {s.section_number && (
                          <span className="source-section"> §{s.section_number}</span>
                        )}
                        <span className="score">{s.score.toFixed(2)}</span>
                      </span>
                    ))}
                  </div>
                )}
                {m.role === "assistant" && !m.refused && m.follow_ups && m.follow_ups.length > 0 && (
                  <div className="follow-ups">
                    <span className="follow-ups-label">You might also ask</span>
                    <div className="follow-ups-grid">
                      {m.follow_ups.map((q, i) => (
                        <button key={i} className="follow-up-btn" onClick={() => sendText(q)}>
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {sending && (
            <div className="msg msg-assistant">
              <div className="msg-avatar" aria-hidden>⚖</div>
              <div className="msg-body">
                <div className="msg-meta"><span className="msg-role">NyayaBot</span></div>
                <div className="msg-content typing">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="composer-wrap">
          <div className="composer-mode-row">
            <button
              type="button"
              className={`mode-tab${!scenarioMode ? " mode-tab--active" : ""}`}
              onClick={() => setScenarioMode(false)}
            >
              Free text
            </button>
            <button
              type="button"
              className={`mode-tab${scenarioMode ? " mode-tab--active" : ""}`}
              onClick={() => setScenarioMode(true)}
            >
              Describe situation
            </button>
          </div>

          {scenarioMode ? (
            <div className="scenario-form">
              <textarea
                className="scenario-textarea"
                rows={3}
                placeholder="What happened? Describe in your own words…"
                value={scenario.what}
                onChange={(e) => setScenario((s) => ({ ...s, what: e.target.value }))}
                disabled={sending}
              />
              <div className="scenario-row">
                <label>Who is involved?</label>
                <select
                  value={scenario.who}
                  onChange={(e) => setScenario((s) => ({ ...s, who: e.target.value }))}
                  disabled={sending}
                >
                  {WHO_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o.charAt(0).toUpperCase() + o.slice(1)}</option>
                  ))}
                </select>
              </div>
              <div className="scenario-row">
                <label>When?</label>
                <input
                  type="text"
                  placeholder="e.g. 3 months ago, Jan 2025"
                  value={scenario.when}
                  onChange={(e) => setScenario((s) => ({ ...s, when: e.target.value }))}
                  disabled={sending}
                />
              </div>
              <div className="scenario-row">
                <label>Where?</label>
                <select
                  value={scenario.where}
                  onChange={(e) => setScenario((s) => ({ ...s, where: e.target.value }))}
                  disabled={sending}
                >
                  {WHERE_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="scenario-row">
                <label>What outcome?</label>
                <select
                  value={scenario.outcome}
                  onChange={(e) => setScenario((s) => ({ ...s, outcome: e.target.value }))}
                  disabled={sending}
                >
                  {OUTCOME_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </div>
              <div className="scenario-actions">
                <button
                  type="button"
                  className="scenario-submit-btn"
                  onClick={sendScenario}
                  disabled={sending || !scenario.what.trim()}
                >
                  {sending ? "…" : "Analyse my situation"}
                </button>
              </div>
            </div>
          ) : (
            <form className="composer" onSubmit={onSubmit}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                className="upload-input-hidden"
                onChange={handleFileChange}
              />
              <button
                type="button"
                className={`upload-btn${sessionDoc ? " upload-btn--has-doc" : ""}`}
                title={sessionDoc ? `Document: ${sessionDoc.display_name}` : "Upload PDF document"}
                onClick={() => !sessionDoc && fileInputRef.current?.click()}
                disabled={uploading || !!sessionDoc}
                aria-label="Upload document"
              >
                {uploading ? "…" : sessionDoc ? "📄" : "⊕"}
              </button>
              <textarea
                ref={textareaRef}
                rows={1}
                placeholder="Ask a legal question…  (Shift+Enter for newline)"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKeyDown}
                disabled={sending}
              />
              <button type="submit" disabled={sending || !input.trim()}>
                {sending ? "…" : "Send"}
              </button>
            </form>
          )}

          {uploadError && (
            <div className="upload-error small">{uploadError}</div>
          )}
          <div className="composer-hint muted small">
            NyayaBot answers strictly from ingested PDFs. Out-of-scope questions will be refused.
          </div>
        </div>
      </main>
    </div>
  );
}
