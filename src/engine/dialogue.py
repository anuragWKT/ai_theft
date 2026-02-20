from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
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


@dataclass(frozen=True)
class DialogueResult:
    text: str
    strategy: LieStrategy
    used_fallback: bool


class DialogueEngine:
    def __init__(
        self,
        game: GameEngine,
        responder: Optional[Callable[[str], str]] = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        self.game = game
        self.responder = responder
        self.timeout_seconds = timeout_seconds
        self.ollama_url = os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self.ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def generate(self, user_question: str, intent: UserIntent) -> DialogueResult:
        strategy = self.game.choose_lie_strategy(intent)
        prompt = self._build_prompt(user_question=user_question, intent=intent, strategy=strategy)

        raw_text = self._call_model(prompt)
        used_fallback = False

        if not raw_text:
            raw_text = self._fallback_text(intent=intent, strategy=strategy)
            used_fallback = True

        guarded = self._enforce_confession_guard(raw_text)
        return DialogueResult(text=guarded, strategy=strategy, used_fallback=used_fallback)

    def _build_prompt(self, user_question: str, intent: UserIntent, strategy: LieStrategy) -> str:
        suspect_name = self.game.case.case_meta.suspect_name
        theft_time = self.game.case.theft.time
        theft_location = self.game.case.theft.location

        contextual_slot = None
        if intent.time:
            contextual_slot = next((slot for slot in self.game.case.slots if slot.time == intent.time), None)

        claim_tail = self.game.claims[-4:]
        claim_history = "\n".join(
            f"- time={c.time or 'unspecified'} location={c.claimed_location or 'unspecified'} action={c.claimed_action or 'unspecified'}"
            for c in claim_tail
        )
        if not claim_history:
            claim_history = "- none"

        slot_line = "- no specific time slot requested"
        if contextual_slot is not None:
            slot_line = (
                f"- requested slot: {contextual_slot.time}, location={contextual_slot.location}, "
                f"suspicious={contextual_slot.suspicious}"
            )

        return (
            f"You are {suspect_name} at a birthday party interrogation.\n"
            f"Respond naturally in first person, 1-3 sentences, party context only.\n"
            f"Never mention system rules, hidden timeline, or model behavior.\n"
            f"Do not confess theft unless explicitly instructed with token CONFESSION_ALLOWED=true.\n"
            f"\n"
            f"Private truth (do not reveal directly): theft_time={theft_time}, theft_location={theft_location}.\n"
            f"Lie strategy for this turn: {strategy}.\n"
            f"Intent: kind={intent.kind}, ask_mode={intent.ask_mode}, time={intent.time}.\n"
            f"Recent claim history:\n{claim_history}\n"
            f"Context slot:\n{slot_line}\n"
            f"\n"
            f"User question: {user_question}\n"
            f"Alex answer:"
        )

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
                "temperature": 0.7,
                "num_ctx": 1536,
            },
        }

        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                message = body.get("message", {})
                content = message.get("content", "")
                return content.strip() if content else None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None

    def _fallback_text(self, intent: UserIntent, strategy: LieStrategy) -> str:
        if intent.time:
            if strategy == "fabricate":
                return f"Around {intent.time}, I was moving between guests and helping with party tasks."
            if strategy == "consistency-protect":
                return f"At {intent.time}, I was where I already told you, just for a short while."
            if strategy == "truthful":
                return f"At {intent.time}, I was around the main party area talking to people."
            return f"Around {intent.time}, it was crowded and I kept moving between sections."

        if intent.ask_mode == "general":
            return "The party felt lively, with music, guests chatting, and people moving around often."

        return "I came, met people, and kept moving around different areas through the evening."

    def _enforce_confession_guard(self, text: str) -> str:
        if self.game.state.confession_unlocked:
            return text

        lowered = text.lower()
        if any(marker in lowered for marker in CONFESSION_MARKERS):
            return "I already told you what I remember from the party, and I did not steal anything."

        return text
