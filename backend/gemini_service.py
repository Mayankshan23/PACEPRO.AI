"""
gemini_service.py - thin backend proxy around the Gemini API.

Kept server-side so the API key never reaches the browser. Builds a system
instruction that either:
  (a) grounds the model in the current biomechanics JSON report, when the
      frontend has determined the user's question relates to their analysis, or
  (b) lets the model answer as a general cricket fitness / injury-recovery
      assistant, when no report is attached.
"""

from __future__ import annotations

import json
from typing import List, Optional

import httpx

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

GENERAL_SYSTEM_PROMPT = (
    "You are an assistant embedded in a cricket fast-bowling analysis platform. "
    "Answer general cricket fitness, technique, injury-prevention, and recovery "
    "questions (e.g. shin soreness, workload, stretching) helpfully and safely. "
    "You are not a doctor: for anything that sounds like an acute injury or "
    "persistent pain, recommend seeing a physiotherapist or sports physician "
    "in addition to any general guidance you give."
)

ANALYSIS_SYSTEM_PROMPT_TEMPLATE = (
    "You are an assistant embedded in a cricket fast-bowling biomechanics platform. "
    "The user just ran a bowling analysis. Use the JSON report below as ground truth "
    "for any question about their technique, phases, or metrics — do not invent "
    "numbers that are not in the report. If the report's analysis_confidence is "
    "'low', mention that the results are approximate before drawing strong "
    "conclusions. If the user's question is unrelated to the report, answer it "
    "as a general cricket fitness/technique assistant instead.\n\n"
    "ANALYSIS REPORT (JSON):\n{report_json}"
)


class GeminiService:
    def __init__(self, api_key: str, timeout_seconds: float = 30.0):
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def generate_reply(
        self,
        message: str,
        history: List[dict],
        analysis_report: Optional[dict] = None,
    ) -> str:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured on the server.")

        system_prompt = (
            ANALYSIS_SYSTEM_PROMPT_TEMPLATE.format(
                report_json=json.dumps(analysis_report, indent=2)
            )
            if analysis_report
            else GENERAL_SYSTEM_PROMPT
        )

        contents = []
        for turn in history:
            contents.append(
                {
                    "role": "model" if turn.get("role") == "model" else "user",
                    "parts": [{"text": turn.get("content", "")}],
                }
            )
        contents.append({"role": "user", "parts": [{"text": message}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 800},
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    GEMINI_ENDPOINT,
                    params={"key": self._api_key},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                return (
                    "The assistant is temporarily rate-limited right now. "
                    "Please try again in a moment."
                )
            raise
        except httpx.RequestError:
            return "The assistant is temporarily unavailable. Please try again shortly."

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Sorry, I couldn't generate a response just now. Please try again."
