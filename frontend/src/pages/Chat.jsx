import { useEffect, useMemo, useRef, useState } from "react";
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
    const { data } = await api.get(`/sessions/${id}/messages`);
    setMessages(data);
  };

  const newSession = async () => {
    const { data } = await api.post("/sessions", {});
    setSessions((s) => [data, ...s]);
    setActiveId(data.id);
    setMessages([]);
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
      const { data } = await api.post("/chat", {
        session_id: sessionId,
        message: text,
      });
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
              {s.title}
            </li>
          ))}
          {sessions.length === 0 && (
            <li className="muted small">No chats yet</li>
          )}
        </ul>
        <div className="sidebar-foot">
          <div className="user-row">
            <div className="avatar">{initials}</div>
            <div className="user-meta">
              <div className="user-email" title={user?.email}>
                {user?.email}
              </div>
              <button className="link" onClick={logout}>
                Sign out
              </button>
            </div>
          </div>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-left">
            <div className="chat-title">
              {activeSession?.title || "New conversation"}
            </div>
            <div className="chat-sub muted small">
              Grounded in your ingested legal documents
            </div>
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
                  <button
                    key={q}
                    className="suggestion"
                    onClick={() => sendText(q)}
                  >
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
                  <span className="msg-role">
                    {m.role === "user" ? "You" : "NyayaBot"}
                  </span>
                  {m.refused && (
                    <span className="badge badge-warn">out of scope</span>
                  )}
                  {m.error && <span className="badge badge-err">error</span>}
                </div>
                <div className="msg-content">
                  {m.role === "assistant" ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {m.content}
                    </ReactMarkdown>
                  ) : (
                    m.content
                  )}
                </div>
                {m.role === "assistant" &&
                  m.sources &&
                  m.sources.length > 0 && (
                    <div className="sources">
                      <span className="sources-label">Sources</span>
                      {m.sources.map((s, i) => (
                        <span
                          key={`${s.source}-${i}`}
                          className="source-chip"
                          title={`similarity ${s.score.toFixed(3)}`}
                        >
                          <span className="dot" />
                          {s.source}
                          <span className="score">{s.score.toFixed(2)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                {m.role === "assistant" &&
                  !m.refused &&
                  m.follow_ups &&
                  m.follow_ups.length > 0 && (
                    <div className="follow-ups">
                      <span className="follow-ups-label">
                        You might also ask
                      </span>
                      <div className="follow-ups-grid">
                        {m.follow_ups.map((q, i) => (
                          <button
                            key={i}
                            className="follow-up-btn"
                            onClick={() => sendText(q)}
                          >
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
              <div className="msg-avatar" aria-hidden>
                ⚖
              </div>
              <div className="msg-body">
                <div className="msg-meta">
                  <span className="msg-role">NyayaBot</span>
                </div>
                <div className="msg-content typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        <form className="composer" onSubmit={onSubmit}>
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
        <div className="composer-hint muted small">
          NyayaBot answers strictly from ingested PDFs. Out-of-scope questions
          will be refused.
        </div>
      </main>
    </div>
  );
}
