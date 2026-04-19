from __future__ import annotations

import json
from datetime import datetime, UTC
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class BoardSnapshot(BaseModel):
    game_id: str
    ply_index: int
    fen: str
    turn: Literal["white", "black"]
    legal_moves: list[str]  # UCI strings
    move_history: list[str]  # UCI strings
    is_check: bool
    is_game_over: bool
    result_if_terminal: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> BoardSnapshot:
        return cls.model_validate_json(data)


class PrimitiveValue(BaseModel):
    primitive_id: str
    name: str
    side: Literal["self", "opponent", "global"]
    value: Any
    normalized_value: float = Field(ge=0.0, le=1.0)
    text_render: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    category: str

    def to_json(self) -> str:
        return self.model_dump_json()


class WeightedPrimitiveState(BaseModel):
    primitive_id: str
    name: str
    category: str
    raw_value: Any
    normalized_value: float
    weight: float
    confidence: float
    weighted_score: float  # = normalized_value * weight * confidence
    importance_label: str  # "critical" | "high" | "medium" | "low" | "negligible"
    policy_message: str

    def to_json(self) -> str:
        return self.model_dump_json()


class CandidateMoveRecord(BaseModel):
    uci: str
    san: str
    internal_rank: int = 0       # 1-based; lower = better; set by ranker
    presentation_index: int = 0  # 0-based; position shown to LLM; set by shuffler
    source: Literal["engine", "heuristic", "random", "hybrid"]
    board_after_fen: str = ""
    primitive_delta_summary: str = ""
    candidate_narrative: str = ""
    engine_eval_before: Optional[float] = None
    engine_eval_after: Optional[float] = None
    risk_flags: list[str] = Field(default_factory=list)
    notes: str = ""
    score_delta: Optional[float] = None  # sum of effective-score deltas across primitives
    primitives_after: list[PrimitiveValue] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json()


class MoveRecord(BaseModel):
    ply_index: int
    uci: str
    san: str
    fen_before: str
    fen_after: str
    player: Literal["self", "opponent"]
    primitives: list[PrimitiveValue] = Field(default_factory=list)
    engine_eval_before: Optional[float] = None  # centipawns, from side-to-move perspective
    engine_eval_after: Optional[float] = None
    centipawn_loss: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self) -> str:
        return self.model_dump_json()


class GameTrace(BaseModel):
    game_id: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = None
    config: dict[str, Any] = Field(default_factory=dict)
    white_player: str
    black_player: str
    initial_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves: list[MoveRecord] = Field(default_factory=list)
    result: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"
    termination: Optional[str] = None  # "checkmate", "stalemate", "draw", "resignation"
    total_plies: int = 0
    white_accuracy: Optional[float] = None
    black_accuracy: Optional[float] = None
    average_centipawn_loss_white: Optional[float] = None
    average_centipawn_loss_black: Optional[float] = None

    @model_validator(mode="after")
    def compute_totals(self) -> GameTrace:
        self.total_plies = len(self.moves)
        return self

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> GameTrace:
        return cls.model_validate_json(data)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> GameTrace:
        with open(path) as f:
            return cls.from_json(f.read())


class ExperimentManifest(BaseModel):
    run_id: str
    config_path: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    game_ids: list[str] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = None
    total_games: int = 0
    completed_games: int = 0
    failed_games: int = 0
    player_type: str = "llm"
    narrative_mode: str = "on"
    candidate_mode: str = "engine_assisted"
    policy_name: str = "balanced"
    opponent_type: str = "stockfish"
    opponent_skill: int = 3
    prompt_version: str = "v1.1"
    random_seed: Optional[int] = None
    notes: str = ""

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> ExperimentManifest:
        with open(path) as f:
            return cls.model_validate_json(f.read())


class InterpreterNarrative(BaseModel):
    situation: str
    alert: str
    priority: str
    directive: str
    raw_output: str
    token_count: int
    prompt_version: str = "interpreter_v1.0"

    def to_json(self) -> str:
        return self.model_dump_json()


class DecisionPromptRecord(BaseModel):
    system_prompt: str
    user_prompt: str
    position_summary: str = ""
    policy_summary: str = ""
    candidate_block: str = ""
    response_schema: str = ""
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.2
    prompt_version: str = "v1.1"
    token_count: int = 0

    def to_json(self) -> str:
        return self.model_dump_json()


class DecisionRecord(BaseModel):
    selected_uci: str
    selected_internal_rank: int
    selected_presentation_index: int
    reasoning_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    fallback_used: bool = False
    is_valid: bool = True

    def to_json(self) -> str:
        return self.model_dump_json()


class MoveTrace(BaseModel):
    game_id: str
    ply_index: int
    board_snapshot: BoardSnapshot
    primitives: list[PrimitiveValue] = Field(default_factory=list)
    weighted_state: list[WeightedPrimitiveState] = Field(default_factory=list)
    candidates: list[CandidateMoveRecord] = Field(default_factory=list)
    prompt_record: Optional[DecisionPromptRecord] = None
    decision_record: Optional[DecisionRecord] = None
    engine_eval_before: Optional[float] = None
    engine_eval_after: Optional[float] = None
    centipawn_loss: Optional[float] = None
    blunder_label: Optional[str] = None  # "blunder", "mistake", "inaccuracy", None
    token_count: int = 0
    position_narrative: str = ""
    position_narrative_word_count: int = 0
    position_narrative_token_count: int = 0
    interpreter_narrative: Optional[InterpreterNarrative] = None
    interpreter_tokens: Optional[int] = None
    total_tokens_this_move: Optional[int] = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
