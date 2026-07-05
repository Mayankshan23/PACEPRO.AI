const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Uploads 1-3 bowling videos and returns the biomechanics report.
 * @param {{front?: File, back?: File, side?: File}} files
 */
export async function analyzeBowling(files) {
  const formData = new FormData();
  Object.entries(files).forEach(([view, file]) => {
    if (file) formData.append(view, file);
  });

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Analysis failed.");
  }
  const report = await res.json();

  // Resolve video URLs to absolute, ready-to-stream paths.
  const videoUrls = {};
  Object.entries(report.video_urls || {}).forEach(([view, path]) => {
    videoUrls[view] = `${API_BASE}${path}`;
  });
  return { ...report, video_urls: videoUrls };
}

/**
 * Sends a chat message. If `analysisReport` is passed, the backend grounds
 * Gemini's answer in that JSON; otherwise it answers as a general assistant.
 */
export async function sendChatMessage(message, history, analysisReport) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      history,
      analysis_report: analysisReport || null,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request failed.");
  }
  const data = await res.json();
  return data.reply;
}
