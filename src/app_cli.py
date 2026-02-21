from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.engine.dialogue import DialogueEngine
from src.engine.game_logic import GameEngine, UserIntent, load_party_case, parse_intent
from src.rag.store import RAGStore


def create_runtime(project_root: Optional[Path] = None) -> tuple[GameEngine, DialogueEngine]:
    root = project_root or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    case = load_party_case(root / "src" / "data" / "story" / "party_case.json")
    game = GameEngine(case)

    rag_store = None
    try:
        rag_store = RAGStore(project_root=root)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rag_store.build(reset=False)
    except Exception:
        rag_store = None

    dialogue = DialogueEngine(game=game, rag_store=rag_store)
    return game, dialogue


def process_turn(user_input: str, game: GameEngine, dialogue: DialogueEngine) -> Dict[str, Any]:
    text = user_input.strip()
    if not text:
        return {"message": "Please ask a timeline, camera, or accusation question.", "finished": False}

    lowered = text.lower()

    if lowered in {"exit", "quit"}:
        return {"message": "Session ended.", "finished": True}

    intent: UserIntent = parse_intent(text)

    if intent.kind in {"help", "status"} or lowered in {"tips", "tip"}:
        return {
            "message": "Command mode is disabled in voice play. Ask timeline, camera, or accuse questions.",
            "finished": False,
        }

    if intent.kind == "camera":
        if not intent.time:
            return {"message": "Please provide a time, for example: camera 9:30 pm", "finished": False}

        event = game.get_camera_event(intent.time)
        if event is None:
            return {"message": f"No camera coverage found for {intent.time}.", "finished": False}

        camera_line = (
            f"Camera {event.time}: {event.location} | observed: {event.observed_action} | "
            f"people: {', '.join(event.observed_people)}"
        )
        return {"message": camera_line, "finished": False}

    if intent.kind == "accuse":
        result = game.evaluate_accusation(intent)
        if result.confession:
            confession_text = dialogue.generate_confession(text)
            return {
                "message": f"{result.message}\nAlex: {confession_text}",
                "finished": True,
                "confession": True,
            }

        return {
            "message": f"{result.message}\nAlex: I did not steal anything.",
            "finished": False,
            "confession": False,
        }

    resolved_intent = game.resolve_followup_time(intent)
    reply = dialogue.generate(user_question=text, intent=resolved_intent)
    inferred_location = game.infer_location_from_text(reply.text)
    claim = game.record_claim(
        raw_text=reply.text,
        time=resolved_intent.time,
        claimed_location=inferred_location,
        claimed_action=None,
    )
    game.evaluate_and_store_contradictions(claim)
    game.note_turn_time(resolved_intent)

    suffix = ""
    if reply.used_fallback:
        suffix = "\n(note: fallback response used)"

    return {
        "message": f"Alex: {reply.text}{suffix}",
        "finished": False,
        "strategy": reply.strategy,
    }


def main() -> None:
    game, dialogue = create_runtime()

    print("=== Birthday Party Interrogation ===")
    print("You are questioning Alex about the missing gold chain.")
    print("Ask timeline questions, use camera checks, then accuse with evidence.")

    while True:
        user_input = input("\nYou: ")
        result = process_turn(user_input=user_input, game=game, dialogue=dialogue)
        print(result["message"])

        if result.get("finished"):
            break


if __name__ == "__main__":
    main()
