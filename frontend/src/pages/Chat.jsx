import { useEffect, useRef, useState } from "react";
import api from "../api/client.js";
import { useAuth } from "../context/AuthContext.jsx";

export default function Chat() {
  const { user, logout } = useAuth();
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef(null);

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
  }, [messages]);

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

  const send = async (e) => {
    e.preventDefault();
    const text = input.trim();
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
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="chat-shell">
      <aside className="sidebar">
        <div className="sidebar-head">
          <button className="new-btn" onClick={newSession}>+ New chat</button>
        </div>
        <ul className="sessions">
          {sessions.map((s) => (
            <li
              key={s.id}
              className={s.id === activeId ? "active" : ""}
              onClick={() => selectSession(s.id)}
            >
              {s.title}
            </li>
          ))}
          {sessions.length === 0 && <li className="muted">No chats yet</li>}
        </ul>
        <div className="sidebar-foot">
          <div className="muted small">{user?.email}</div>
          <button className="link" onClick={logout}>Sign out</button>
        </div>
      </aside>

      <main className="chat-main">
        <div className="messages" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="empty muted">Ask a question about the ingested legal documents.</div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`msg ${m.role}${m.refused ? " refused" : ""}`}>
              <div className="role">{m.role === "user" ? "You" : "NyayaBot"}</div>
              <div className="content">{m.content}</div>
            </div>
          ))}
          {sending && <div className="msg assistant"><div className="role">NyayaBot</div><div className="content muted">Thinking…</div></div>}
        </div>
        <form className="composer" onSubmit={send}>
          <input
            placeholder="Ask a legal question…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
          />
          <button type="submit" disabled={sending || !input.trim()}>Send</button>
        </form>
      </main>
    </div>
  );
}
