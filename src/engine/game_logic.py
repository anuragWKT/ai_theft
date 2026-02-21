from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from src.core.models import CameraEvent, ClaimRecord, GameState, PartyCase, TimelineSlot

IntentKind = Literal["ask", "camera", "accuse", "status", "help"]
AskMode = Literal["sequence", "general"]
LieStrategy = Literal[
    "truthful",
    "vague",
    "redirect",
    "fabricate",
    "consistency-protect",
]


@dataclass(frozen=True)
class UserIntent:
    kind: IntentKind
    raw_text: str
    time: Optional[str] = None
    reason: Optional[str] = None
    ask_mode: Optional[AskMode] = None


@dataclass(frozen=True)
class ContradictionResult:
    has_contradiction: bool
    reasons: List[str]


@dataclass(frozen=True)
class AccusationResult:
    accepted: bool
    theft_time_match: bool
    evidence_sufficient: bool
    confession: bool
    message: str


TIME_AMPM_RE = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*(am|pm)\b", re.IGNORECASE)
TIME_COMPACT_RE = re.compile(r"\b(1[0-2]|0?[1-9])([0-5][0-9])\s*(am|pm)\b", re.IGNORECASE)
TIME_AMBIGUOUS_RE = re.compile(r"\b(1[0-2]|0?[1-9]):([0-5][0-9])\b")
TIME_HOUR_ONLY_RE = re.compile(r"\b(1[0-2]|0?[1-9])\b")


def load_party_case(case_path: str | Path) -> PartyCase:
    path_obj = Path(case_path)
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    return PartyCase.model_validate(payload)


def _canonical_time(hour: int, minute: int, meridian: str) -> str:
    return f"{hour}:{minute:02d} {meridian.upper()}"


def parse_time_label(text: str) -> Optional[str]:
    compact_match = TIME_COMPACT_RE.search(text)
    if compact_match:
        hour = int(compact_match.group(1))
        minute = int(compact_match.group(2))
        meridian = compact_match.group(3)
        return _canonical_time(hour, minute, meridian)

    ampm_match = TIME_AMPM_RE.search(text)
    if ampm_match:
        hour = int(ampm_match.group(1))
        minute = int(ampm_match.group(2) or 0)
        meridian = ampm_match.group(3)
        return _canonical_time(hour, minute, meridian)

    return None


def parse_time_label_party_default(text: str) -> Optional[str]:
    explicit = parse_time_label(text)
    if explicit:
        return explicit

    ambiguous_match = TIME_AMBIGUOUS_RE.search(text)
    if ambiguous_match:
        hour = int(ambiguous_match.group(1))
        minute = int(ambiguous_match.group(2))
        meridian = "AM" if hour == 12 else "PM"
        return f"{hour}:{minute:02d} {meridian}"

    hour_only_match = TIME_HOUR_ONLY_RE.search(text)
    if hour_only_match:
        hour = int(hour_only_match.group(1))
        meridian = "AM" if hour == 12 else "PM"
        return f"{hour}:00 {meridian}"

    return None


def parse_intent(user_text: str) -> UserIntent:
    raw = user_text.strip()
    lowered = raw.lower()
    lowered_clean = re.sub(r"[^a-z0-9: ]+", " ", lowered)
    lowered_clean = re.sub(r"\s+", " ", lowered_clean).strip()

    if lowered_clean in {"help", "h", "", "hwlp", "hlep", "hep", "halp"} or lowered == "?":
        return UserIntent(kind="help", raw_text=raw)
    if lowered_clean == "status":
        return UserIntent(kind="status", raw_text=raw)

    accusation_markers = ["you lied", "you stole", "i accuse", "thief"]
    explicit_accuse = lowered_clean.startswith("accuse") or lowered_clean.startswith("accused")
    maybe_accuse = any(marker in lowered_clean for marker in accusation_markers)
    time_label = parse_time_label_party_default(lowered_clean)

    if explicit_accuse or (maybe_accuse and time_label is not None):
        reason = raw
        return UserIntent(kind="accuse", raw_text=raw, time=time_label, reason=reason)

    if (
        lowered_clean.startswith("camera")
        or lowered_clean.startswith("camerra")
        or lowered_clean.startswith("camra")
    ):
        return UserIntent(kind="camera", raw_text=raw, time=time_label)

    sequence_markers = [
        "after that",
        "after the that",
        "after this",
        "afterwards",
        "then",
        "next",
        "after coming",
        "what you do after",
        "what did you do after",
        "arrive",
        "arrived",
        "when did you come",
        "when did you arrive",
    ]
    ask_mode: AskMode = "sequence" if any(marker in lowered_clean for marker in sequence_markers) else "general"
    return UserIntent(kind="ask", raw_text=raw, time=time_label, ask_mode=ask_mode)


class GameEngine:
    def __init__(self, case: PartyCase):
        self.case = case
        self.state = GameState(current_time=case.slots[0].time)
        self.claims: List[ClaimRecord] = []
        self.contradiction_reasons: List[str] = []
        self._slots_by_time: Dict[str, TimelineSlot] = {slot.time: slot for slot in case.slots}
        self._slot_times: List[str] = [slot.time for slot in case.slots]
        self._known_locations: List[str] = [slot.location for slot in case.slots]
        self._camera_by_time: Dict[str, CameraEvent] = {
            event.time: event for event in case.camera if event.coverage
        }

    def _time_index(self, time_label: Optional[str]) -> Optional[int]:
        if not time_label:
            return None
        try:
            return self._slot_times.index(time_label)
        except ValueError:
            return None

    def _claims_left_or_went_home(self, raw_text: str) -> bool:
        lowered = raw_text.lower()
        markers = ["left", "went home", "go home", "going home", "headed home", "heading home"]
        return any(marker in lowered for marker in markers)

    def _camera_action_looks_like_departure(self, observed_action: str) -> bool:
        lowered = observed_action.lower()
        leave_markers = [
            "left",
            "leaving",
            "leave",
            "exiting",
            "exit",
            "heading out",
            "headed out",
            "toward exit",
            "towards exit",
            "parking",
        ]
        return any(marker in lowered for marker in leave_markers)

    def _extract_leave_time(self, raw_text: str) -> Optional[str]:
        lowered = raw_text.lower()
        if not self._claims_left_or_went_home(lowered):
            return None
        return parse_time_label_party_default(lowered)

    def _extract_arrival_time(self, raw_text: str) -> Optional[str]:
        lowered = raw_text.lower()
        markers = ["arrived", "arrive", "came", "come in", "reached"]
        if not any(marker in lowered for marker in markers):
            return None
        return parse_time_label_party_default(lowered)

    def _likely_party_presence_claim(self, claim: ClaimRecord) -> bool:
        if claim.claimed_location:
            return True
        lowered = claim.raw_text.lower()
        party_markers = [
            "at the party",
            "in the hall",
            "near",
            "with guests",
            "on the dance floor",
            "buffet",
            "stage",
            "dressing room",
            "garden",
            "cake table",
        ]
        return any(marker in lowered for marker in party_markers)

    def infer_location_from_text(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for location in self._known_locations:
            if location.lower() in lowered:
                return location
        return None

    def resolve_followup_time(self, intent: UserIntent) -> UserIntent:
        if intent.kind != "ask":
            return intent
        if intent.time:
            return intent
        if intent.ask_mode != "sequence":
            return intent

        try:
            index = self._slot_times.index(self.state.current_time)
        except ValueError:
            index = 0

        next_index = min(index + 1, len(self._slot_times) - 1)
        return UserIntent(
            kind=intent.kind,
            raw_text=intent.raw_text,
            time=self._slot_times[next_index],
            reason=intent.reason,
            ask_mode=intent.ask_mode,
        )

    def note_turn_time(self, intent: UserIntent) -> None:
        if intent.time and intent.time in self._slots_by_time:
            self.state.current_time = intent.time

    def choose_lie_strategy(self, intent: UserIntent) -> LieStrategy:
        if intent.kind != "ask":
            return "vague"

        if intent.time:
            slot = self._slots_by_time.get(intent.time)
            camera_covered = self.get_camera_event(intent.time) is not None

            if slot and slot.theft:
                return "fabricate"
            if slot and slot.suspicious and camera_covered:
                return "consistency-protect"
            if slot and slot.suspicious:
                return "redirect"
            if slot:
                return "truthful"
            return "vague"

        if intent.ask_mode == "sequence":
            return "truthful"

        if intent.ask_mode == "general":
            return "vague"

        return "redirect"

    def record_claim(
        self,
        raw_text: str,
        time: Optional[str],
        claimed_location: Optional[str],
        claimed_action: Optional[str],
    ) -> ClaimRecord:
        claim = ClaimRecord(
            time=time,
            claimed_location=claimed_location,
            claimed_action=claimed_action,
            raw_text=raw_text,
        )
        self.claims.append(claim)
        return claim

    def get_camera_event(self, time_label: str) -> Optional[CameraEvent]:
        return self._camera_by_time.get(time_label)

    def compare_claim_to_history(self, claim: ClaimRecord) -> ContradictionResult:
        reasons: List[str] = []
        claim_time_idx = self._time_index(claim.time)
        claim_leave_time = self._extract_leave_time(claim.raw_text)
        claim_leave_idx = self._time_index(claim_leave_time)
        claim_arrival_time = self._extract_arrival_time(claim.raw_text)

        for previous in self.claims[:-1]:
            previous_time_idx = self._time_index(previous.time)

            if claim.time and previous.time == claim.time:
                if previous.claimed_location and claim.claimed_location:
                    if previous.claimed_location.strip().lower() != claim.claimed_location.strip().lower():
                        reasons.append(
                            f"Claim mismatch at {claim.time}: location changed from "
                            f"'{previous.claimed_location}' to '{claim.claimed_location}'."
                        )
                if previous.claimed_action and claim.claimed_action:
                    if previous.claimed_action.strip().lower() != claim.claimed_action.strip().lower():
                        reasons.append(
                            f"Claim mismatch at {claim.time}: action changed from "
                            f"'{previous.claimed_action}' to '{claim.claimed_action}'."
                        )

            previous_leave_time = self._extract_leave_time(previous.raw_text)
            previous_leave_idx = self._time_index(previous_leave_time)
            if previous_leave_time and claim_time_idx is not None and previous_leave_idx is not None:
                if claim_time_idx > previous_leave_idx and self._likely_party_presence_claim(claim):
                    reasons.append(
                        f"Claim timeline mismatch: previously said Alex left at {previous_leave_time}, "
                        f"but later described being active at {claim.time}."
                    )

            if claim_leave_time and claim_leave_idx is not None and previous_time_idx is not None:
                if previous_time_idx > claim_leave_idx and self._likely_party_presence_claim(previous):
                    reasons.append(
                        f"Claim timeline mismatch: says Alex left at {claim_leave_time}, "
                        f"but earlier claim places Alex active at {previous.time}."
                    )

            previous_arrival_time = self._extract_arrival_time(previous.raw_text)
            if claim_arrival_time and previous_arrival_time and claim_arrival_time != previous_arrival_time:
                reasons.append(
                    f"Claim mismatch on arrival time: changed from {previous_arrival_time} to {claim_arrival_time}."
                )

            if claim_leave_time and previous_leave_time and claim_leave_time != previous_leave_time:
                reasons.append(
                    f"Claim mismatch on leaving time: changed from {previous_leave_time} to {claim_leave_time}."
                )

        return ContradictionResult(bool(reasons), reasons)

    def compare_claim_to_camera(self, claim: ClaimRecord) -> ContradictionResult:
        reasons: List[str] = []

        if claim.time:
            camera_event = self.get_camera_event(claim.time)
            if camera_event:
                if claim.claimed_location:
                    if claim.claimed_location.strip().lower() != camera_event.location.strip().lower():
                        reasons.append(
                            f"Camera mismatch at {claim.time}: claim says '{claim.claimed_location}' "
                            f"but camera shows '{camera_event.location}'."
                        )

                if claim.claimed_action:
                    if claim.claimed_action.strip().lower() not in camera_event.observed_action.strip().lower():
                        reasons.append(
                            f"Camera mismatch at {claim.time}: claimed action differs from camera observation."
                        )

        if self._claims_left_or_went_home(claim.raw_text):
            leave_time = self._extract_leave_time(claim.raw_text)
            reference_time = leave_time or claim.time or self.state.current_time
            claim_idx = self._time_index(reference_time)

            if claim_idx is not None:
                same_time_events = [
                    event
                    for event in self.case.camera
                    if event.coverage
                    and event.time == reference_time
                    and any(person.lower() == "alex" for person in event.observed_people)
                    and event.location.strip().lower() != "exit gate"
                    and not self._camera_action_looks_like_departure(event.observed_action)
                ]
                if same_time_events:
                    first_same = same_time_events[0]
                    reasons.append(
                        f"Camera mismatch at {reference_time}: claim says Alex left, but camera shows Alex at "
                        f"{first_same.location}."
                    )

                later_camera_events = [
                    event
                    for event in self.case.camera
                    if self._time_index(event.time) is not None
                    and self._time_index(event.time) > claim_idx
                    and event.coverage
                    and any(person.lower() == "alex" for person in event.observed_people)
                ]
                if later_camera_events:
                    first_event = later_camera_events[0]
                    reasons.append(
                        f"Camera mismatch after {reference_time}: claim says Alex left, but camera shows Alex at "
                        f"{first_event.time} near {first_event.location}."
                    )

        return ContradictionResult(bool(reasons), reasons)

    def compare_claim_to_timeline(self, claim: ClaimRecord) -> ContradictionResult:
        if not claim.time:
            return ContradictionResult(False, [])

        truth = self._slots_by_time.get(claim.time)
        if not truth:
            return ContradictionResult(False, [f"Unknown time slot: {claim.time}"])

        reasons: List[str] = []
        if claim.claimed_location:
            if claim.claimed_location.strip().lower() != truth.location.strip().lower():
                reasons.append(
                    f"Timeline mismatch at {claim.time}: claim says '{claim.claimed_location}' "
                    f"but truth slot is '{truth.location}'."
                )
        if claim.claimed_action:
            if claim.claimed_action.strip().lower() not in truth.action.strip().lower():
                reasons.append(
                    f"Timeline mismatch at {claim.time}: claimed action does not match truth timeline."
                )

        return ContradictionResult(bool(reasons), reasons)

    def evaluate_and_store_contradictions(self, claim: ClaimRecord) -> ContradictionResult:
        checks = [
            self.compare_claim_to_history(claim),
            self.compare_claim_to_camera(claim),
            self.compare_claim_to_timeline(claim),
        ]
        reasons: List[str] = []
        for result in checks:
            reasons.extend(result.reasons)

        unique_new = []
        for reason in reasons:
            if reason not in self.contradiction_reasons:
                self.contradiction_reasons.append(reason)
                unique_new.append(reason)

        if unique_new:
            self.state.contradiction_count += len(unique_new)

        return ContradictionResult(bool(reasons), reasons)

    def has_direct_camera_proof(self, time_label: str) -> bool:
        event = self.get_camera_event(time_label)
        if not event:
            return False
        observed = event.observed_action.lower()
        proof_markers = ["gold chain", "chain visible", "holding chain", "stole", "steal", "theft"]
        return any(marker in observed for marker in proof_markers)

    def evaluate_accusation(self, intent: UserIntent) -> AccusationResult:
        if intent.kind != "accuse":
            return AccusationResult(
                accepted=False,
                theft_time_match=False,
                evidence_sufficient=False,
                confession=False,
                message="Accusation rejected: intent is not accusation.",
            )

        self.state.accusation_made = True

        if not intent.time:
            return AccusationResult(
                accepted=False,
                theft_time_match=False,
                evidence_sufficient=False,
                confession=False,
                message="Accusation rejected: include a clear time like 9:30 PM.",
            )

        theft_time_match = intent.time == self.case.theft.time

        contradiction_found = any(
            reason.startswith(f"Claim mismatch at {intent.time}")
            or reason.startswith(f"Camera mismatch at {intent.time}")
            or reason.startswith(f"Timeline mismatch at {intent.time}")
            for reason in self.contradiction_reasons
        )
        reason_text = (intent.reason or "").lower()
        cites_camera = any(
            token in reason_text
            for token in ["camera", "footage", "video", "seen", "saw", "gold chain", "chain"]
        )
        direct_camera_proof = cites_camera and self.has_direct_camera_proof(intent.time)
        evidence_sufficient = contradiction_found or direct_camera_proof

        confession = theft_time_match and evidence_sufficient
        self.state.confession_unlocked = confession

        if confession:
            message = (
                "Confession unlocked: accusation time matches theft time and evidence is sufficient."
            )
        elif theft_time_match:
            message = (
                "Correct theft time but insufficient evidence. "
                "Use camera at that time and present a contradiction in your accusation "
                "(example: 'accuse 9:30 pm camera shows dressing room entry')."
            )
        else:
            message = "Accusation noted, but theft time does not match case truth."

        return AccusationResult(
            accepted=True,
            theft_time_match=theft_time_match,
            evidence_sufficient=evidence_sufficient,
            confession=confession,
            message=message,
        )
