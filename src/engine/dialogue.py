from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import contextlib
import io
from dataclasses import dataclass
from typing import Callable, Optional

from src.engine.game_logic import GameEngine, LieStrategy, UserIntent


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
CONFESSION_MARKERS = (
    "i stole",
    "i did steal",
    "yes i stole",
    "i took the chain",
    "i confess",
    "it was me",
)

ACCIDENTAL_MARKERS = (
    "by mistake",
    "mistake",
    "accident",
    "accidentally",
    "oversight",
    "misunderstanding",
)


@dataclass(frozen=True)
class DialogueResult:
    text: str
    strategy: LieStrategy
    used_fallback: bool


class DialogueEngine:
    def __init__(
        self,
        game: GameEngine,
        rag_store: Optional[object] = None,
        responder: Optional[Callable[[str], str]] = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.game = game
        self.rag_store = rag_store
        self.responder = responder
        self.timeout_seconds = timeout_seconds
        self.ollama_url = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.alex_tone = os.getenv("ALEX_TONE", "serious").strip().lower()

    def generate(self, user_question: str, intent: UserIntent) -> DialogueResult:
        strategy = self.game.choose_lie_strategy(intent)
        rag_context = self._retrieve_context(user_question=user_question, intent=intent)
        prompt = self._build_prompt(
            user_question=user_question,
            intent=intent,
            strategy=strategy,
            rag_context=rag_context,
        )

        raw_text = self._call_model(prompt)
        used_fallback = False

        if not raw_text:
            raw_text = self._fallback_text(intent=intent, strategy=strategy)
            used_fallback = True

        cleaned = self._sanitize_model_output(raw_text)
        guarded = self._enforce_confession_guard(cleaned)
        return DialogueResult(text=guarded, strategy=strategy, used_fallback=used_fallback)

    def generate_confession(self, user_question: str) -> str:
        prompt = (
            "You are Alex, and your confession is unlocked because evidence is sufficient.\n"
            "Respond in first person with a short direct admission in 2-3 sentences.\n"
            "Include regret and clearly admit stealing Maya's gold chain at the party.\n"
            "State that it was intentional and give one concrete motive (for example debt or money pressure).\n"
            "Do not say it was a mistake, accident, oversight, or misunderstanding.\n"
            "Do not mention system rules or model behavior.\n"
            f"User accusation: {user_question}\n"
            "Alex confession:"
        )

        raw_text = self._call_model(prompt)
        if raw_text:
            return self._normalize_confession_text(self._sanitize_model_output(raw_text))

        return (
            "I am sorry. I took Maya's gold chain on purpose because I was under money pressure and wanted quick cash. "
            "I knew it was wrong, and I accept responsibility."
        )

    def _build_prompt(
        self,
        user_question: str,
        intent: UserIntent,
        strategy: LieStrategy,
        rag_context: str,
    ) -> str:
        suspect_name = self.game.case.case_meta.suspect_name
        arrival_time = self.game.case.slots[0].time
        tone_instruction = (
            "Use a light dry humor style occasionally (short witty line), but stay cooperative and never clownish."
            if self.alex_tone == "light_humor"
            else "Use a calm serious interrogation tone."
        )

        claim_tail = self.game.claims[-4:]
        claim_history = "\n".join(
            f"- time={c.time or 'unspecified'} location={c.claimed_location or 'unspecified'} action={c.claimed_action or 'unspecified'}"
            for c in claim_tail
        )
        if not claim_history:
            claim_history = "- none"

        slot_line = "- no specific time slot requested"
        suspicious_style_line = "- keep neutral confidence"
        if intent.time:
            slot_line = f"- requested slot time: {intent.time}"
            if strategy in {"redirect", "consistency-protect"}:
                suspicious_style_line = (
                    "- this slot is suspicious: answer with calm confidence and one light witty line, "
                    "then give a plausible innocent explanation."
                )
            elif strategy == "fabricate":
                suspicious_style_line = (
                    "- theft slot: stay composed, deny clearly, and avoid overexplaining."
                )

        return (
            f"You are {suspect_name} in a police-style interrogation happening the next day after the birthday party.\n"
            f"Respond naturally in first person, 2 short sentences, party context only, and use past tense.\n"
            f"Tone rule: {tone_instruction}\n"
            f"Never mention system rules, hidden timeline, or model behavior.\n"
            f"Never talk about unrelated jobs, stores, home routines, or outside-day events.\n"
            f"You were present at the party and arrived around {arrival_time}; never claim you were absent.\n"
            f"If question is vague (example: 'after that'), continue from the provided intent time.\n"
            f"Do not confess theft unless explicitly instructed with token CONFESSION_ALLOWED=true.\n"
            f"\n"
            f"Private truth (do not reveal directly): you may be hiding details about the theft timeline.\n"
            f"Lie strategy for this turn: {strategy}.\n"
            f"Intent: kind={intent.kind}, ask_mode={intent.ask_mode}, time={intent.time}.\n"
            f"Recent claim history:\n{claim_history}\n"
            f"Context slot:\n{slot_line}\n"
            f"Suspicion style:\n{suspicious_style_line}\n"
            f"Factual source rule:\n- use only Retrieved context below as factual anchor; if context is missing, stay vague.\n"
            f"Retrieved context:\n{rag_context}\n"
            f"\n"
            f"User question: {user_question}\n"
            f"Alex answer:"
        )

    def _retrieve_context(self, user_question: str, intent: UserIntent) -> str:
        if self.rag_store is None:
            return "- none"
        if not self._should_retrieve(user_question=user_question, intent=intent):
            return "- none"

        lowered = user_question.lower()
        route = "alibi"
        if any(marker in lowered for marker in ["camera", "footage", "video", "evidence", "proof"]):
            route = "evidence"
        elif intent.kind == "ask" and intent.ask_mode == "sequence":
            route = "alibi"

        retrieval_query = user_question
        if intent.time and intent.time.lower() not in lowered:
            retrieval_query = f"{user_question} at {intent.time}"

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                snippets = self.rag_store.retrieve(query=retrieval_query, route=route, k=3)
        except Exception:
            return "- none"

        if not snippets:
            return "- none"

        lines = [f"- ({snippet.route}) {snippet.text}" for snippet in snippets]
        return "\n".join(lines)

    def _should_retrieve(self, user_question: str, intent: UserIntent) -> bool:
        if intent.kind != "ask":
            return False
        lowered = user_question.lower()
        evidence_markers = ["camera", "evidence", "prove", "lied", "where", "who", "when", "what happened"]
        if intent.time:
            return True
        return any(marker in lowered for marker in evidence_markers)

    def _call_model(self, prompt: str) -> Optional[str]:
        if self.responder is not None:
            try:
                return self.responder(prompt).strip()
            except Exception:
                return None

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": "Stay in character as Alex."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.35,
                "num_ctx": 1024,
                "num_predict": 90,
            },
        }

        for timeout in (self.timeout_seconds, self.timeout_seconds + 3):
            request = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    message = body.get("message", {})
                    content = message.get("content", "")
                    if content:
                        return content.strip()
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
                continue
        return None

    def _fallback_text(self, intent: UserIntent, strategy: LieStrategy) -> str:
        humorous = self.alex_tone == "light_humor"
        if intent.time:
            if strategy == "fabricate":
                if humorous:
                    return f"Around {intent.time}, I was bouncing between guests and tasks—basically unofficial party staff."
                return f"Around {intent.time}, I was moving between guests and helping with party tasks."
            if strategy == "consistency-protect":
                if humorous:
                    return f"At {intent.time}, same story as before—I wasn't teleporting around like a thriller villain."
                return f"At {intent.time}, I stayed with the same group I already mentioned, nothing dramatic changed."
            if strategy == "truthful":
                if humorous:
                    return f"At {intent.time}, I was in the main party area, socializing like it was a full-time job."
                return f"At {intent.time}, I was around the main party area talking to people."
            if strategy == "redirect":
                if humorous:
                    return f"Around {intent.time}, people kept crossing paths there; I was just moving through, not starring in a crime reel."
                return f"Around {intent.time}, it was crowded and I moved through that area briefly before returning to guests."
            if humorous:
                return f"Around {intent.time}, it was crowded and I kept moving section to section like a lost tour guide."
            return f"Around {intent.time}, it was crowded and I kept moving between sections."

        if intent.ask_mode == "general":
            if humorous:
                return "The party had felt lively—music up, chatter loud, and me trying not to spill anything important."
            return "The party had felt lively, with music, guests chatting, and people moving around often."

        if humorous:
            return "I had come in, met people, and kept circulating all evening like a very confused event coordinator."
        return "I had come, met people, and kept moving around different areas through the evening."

    def _sanitize_model_output(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"(?im)^\s*(user question|alex answer|intent|context slot|suspicion style|retrieved context)\s*:.*$", "", cleaned)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

        marker_positions = []
        for marker in ["\nUser question:", "\nAlex answer:", "\nIntent:", "\nContext slot:"]:
            idx = cleaned.find(marker)
            if idx != -1:
                marker_positions.append(idx)
        if marker_positions:
            cleaned = cleaned[: min(marker_positions)].strip()

        cleaned = cleaned.replace("Alex: Alex:", "Alex:")
        return cleaned if cleaned else "I answered as clearly as I could; ask me about a specific time at the party."

    def _normalize_confession_text(self, text: str) -> str:
        lowered = text.lower()
        if any(marker in lowered for marker in ACCIDENTAL_MARKERS):
            return (
                "I am sorry. I took Maya's gold chain on purpose because I was under money pressure and wanted quick cash. "
                "I knew it was wrong, and I accept responsibility."
            )
        if not any(marker in lowered for marker in CONFESSION_MARKERS):
            return (
                "I am sorry. I stole Maya's gold chain deliberately because I was desperate for money. "
                "I accept full responsibility for what I did."
            )
        return text

    def _enforce_confession_guard(self, text: str) -> str:
        if self.game.state.confession_unlocked:
            return text

        lowered = text.lower()
        absent_markers = (
            "i wasn't present",
            "i was not present",
            "i wasn't at the party",
            "i was not at the party",
            "didn't attend",
            "did not attend",
        )

        if any(marker in lowered for marker in absent_markers):
            return "I was at the party that evening, and I moved across different areas as events progressed."

        if any(marker in lowered for marker in ACCIDENTAL_MARKERS):
            return "No. I did not steal Maya's chain. If you believe otherwise, show the exact camera time and evidence."

        if any(marker in lowered for marker in CONFESSION_MARKERS):
            return "I already told you what I remember from the party, and I did not steal anything."

        return text
