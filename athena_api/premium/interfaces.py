from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PredictionService(Protocol):
    def get_prediction(self, prediction_id: str) -> dict: ...


class OpportunityRankingService(Protocol):
    def rank(self, prediction_ids: list[str]) -> list[dict]: ...


class RiskService(Protocol):
    def approve(self, opportunity: dict, portfolio: dict) -> bool: ...


class ExecutionService(Protocol):
    def submit(self, order: dict) -> dict: ...


class PositionMonitoringService(Protocol):
    def positions(self) -> list[dict]: ...


class AuditService(Protocol):
    def record(self, event: dict) -> None: ...


@dataclass(frozen=True)
class TradingDisabled:
    """Hard stop used until an approved compliance and production cutover."""

    reason: str = "Live trading is not enabled in Athena Baseball."

    def submit(self, order: dict) -> dict:
        raise PermissionError(self.reason)
