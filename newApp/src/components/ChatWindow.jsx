import { useEffect, useRef, useState } from "react";
import { Send, MessageCircle, X, Sparkles } from "lucide-react";
import { sendChatMessage } from "../services/api";

// Lightweight client-side heuristic: decide whether to attach the analysis
// report as grounding context. This is just a UX affordance — the backend
// system prompt also instructs Gemini to fall back to general advice if the
// question turns out to be unrelated, so a false positive here is harmless.
const ANALYSIS_KEYWORDS = [
  "report", "analysis", "metric", "phase", "knee", "elbow", "trunk",
  "release", "flexion", "angle", "score", "confidence", "front foot",
  "back foot", "run-up", "run up", "cadence", "arm path",
];

function looksLikeAnalysisQuestion(text) {
  const lower = text.toLowerCase();
  return ANALYSIS_KEYWORDS.some((kw) => lower.includes(kw));
}

export default function ChatWindow({ report }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "model",
      content:
        "Hi — I'm your bowling assistant. Ask me about your analysis, or anything about technique, workload, or recovery.",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, open]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

    const nextMessages = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");
    setSending(true);

    const contextReport = report && looksLikeAnalysisQuestion(text) ? report : null;

    try {
      const reply = await sendChatMessage(text, nextMessages.slice(0, -1), contextReport);
      setMessages((prev) => [...prev, { role: "model", content: reply }]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "model", content: `Sorry, something went wrong: ${e.message}` },
      ]);
    } finally {
      setSending(false);
    }
  };

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 flex items-center gap-2 rounded-full bg-[#4C7A5E] text-white px-4 py-3 shadow-lg hover:bg-[#4C7A5E]/85 transition-colors"
      >
        <MessageCircle size={18} />
        <span className="font-condensed uppercase tracking-wide text-sm">Ask Gemini</span>
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 w-96 max-w-[calc(100vw-3rem)] h-[520px] rounded-lg border border-white/10 bg-[#12161C] shadow-2xl flex flex-col overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/10 bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <Sparkles size={16} className="text-[#4C7A5E]" />
          <p className="font-condensed uppercase tracking-wide text-white text-sm">
            Gemini Assistant
          </p>
        </div>
        <button onClick={() => setOpen(false)} className="text-white/40 hover:text-white">
          <X size={16} />
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.map((m, i) => (
          <div
            key={i}
            className={[
              "max-w-[85%] rounded-md px-3 py-2 text-sm leading-relaxed",
              m.role === "user"
                ? "ml-auto bg-[#B33A3A]/20 text-white"
                : "mr-auto bg-white/5 text-white/85",
            ].join(" ")}
          >
            {m.content}
          </div>
        ))}
        {sending && (
          <div className="mr-auto bg-white/5 text-white/50 rounded-md px-3 py-2 text-sm">
            thinking…
          </div>
        )}
      </div>

      <div className="border-t border-white/10 p-3 flex items-center gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder={report ? "Ask about your report, or anything else…" : "Ask a general question…"}
          className="flex-1 bg-white/5 rounded-md px-3 py-2 text-sm text-white placeholder:text-white/30 outline-none focus:ring-1 focus:ring-[#4C7A5E]"
        />
        <button
          onClick={handleSend}
          disabled={sending || !input.trim()}
          className="rounded-md bg-[#4C7A5E] disabled:bg-white/10 disabled:text-white/30 text-white p-2"
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
