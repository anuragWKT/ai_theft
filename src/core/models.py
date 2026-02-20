from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


TIME_LABEL_PATTERN = r"^(1[0-2]|[1-9]):[0-5][0-9] (AM|PM)$"


class CaseMeta(BaseModel):
    title: str
    date: str
    venue: str
    victim_name: str
    suspect_name: str = "Alex"


class TimelineSlot(BaseModel):
    time: str = Field(pattern=TIME_LABEL_PATTERN)
    location: str
    action: str
    theft: bool = False
    suspicious: bool = False
    witnesses: List[str] = Field(default_factory=list)


class CameraEvent(BaseModel):
    time: str = Field(pattern=TIME_LABEL_PATTERN)
    location: str
    observed_action: str
    observed_people: List[str] = Field(default_factory=list)
    coverage: bool = True


class TheftInfo(BaseModel):
    time: str = Field(pattern=TIME_LABEL_PATTERN)
    stolen_item: str
    location: str


class ClaimRecord(BaseModel):
    time: Optional[str] = Field(default=None, pattern=TIME_LABEL_PATTERN)
    claimed_location: Optional[str] = None
    claimed_action: Optional[str] = None
    raw_text: str


class GameState(BaseModel):
    current_time: str = Field(default="5:00 PM", pattern=TIME_LABEL_PATTERN)
    suspicion_score: int = 0
    contradiction_count: int = 0
    accusation_made: bool = False
    confession_unlocked: bool = False


class PartyCase(BaseModel):
    case_meta: CaseMeta
    slots: List[TimelineSlot]
    camera: List[CameraEvent]
    theft: TheftInfo

    @model_validator(mode="after")
    def validate_theft_consistency(self) -> "PartyCase":
        theft_slots = [slot for slot in self.slots if slot.theft]
        if len(theft_slots) != 1:
            raise ValueError("Exactly one timeline slot must be marked as theft=true")

        slot_time = theft_slots[0].time
        if self.theft.time != slot_time:
            raise ValueError("theft.time must match the timeline slot where theft=true")

        return self
