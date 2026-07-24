from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from athena_api.agent import answer_question
from athena_api.auth import AuthUser, current_user
from athena_api.database import Base, engine, get_db
from athena_api.ledger_service import create_revision, prediction_tracks, resolve_prediction
from athena_api.models import Follow, Prediction, UserProfile
from athena_api.output_catalog import catalog_payload, resolve_game_outputs
from athena_api.presentation import group_games, prediction_payload
from athena_api.schemas import (
    AgentAnswer,
    AgentQuestion,
    FollowCreate,
    GameResolution,
    PredictionCreate,
    PredictionRead,
    PredictionResolve,
    ProfileUpdate,
)
from athena_api.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="Athena Baseball API",
    version="1.1.0",
    description="Versioned, evidence-grounded MLB prediction API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health/live", tags=["health"])
def live() -> dict:
    return {"status": "ok", "service": "athena-api"}


@app.get("/health/ready", tags=["health"])
def ready(db: Session = Depends(get_db)) -> dict:
    db.execute(text("SELECT 1"))
    return {"status": "ready", "database": "connected"}


@app.get(f"{settings.api_prefix}/predictions", tags=["predictions"])
def list_predictions(
    game_id: str | None = None,
    category: str | None = None,
    statistic: str | None = None,
    headline_only: bool = True,
    limit: int = Query(default=250, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = select(Prediction)
    if game_id:
        query = query.where(Prediction.game_id == game_id)
    if category:
        query = query.where(Prediction.category == category)
    if statistic:
        query = query.where(Prediction.statistic == statistic)
    if headline_only:
        query = query.where(Prediction.is_headline.is_(True))
    rows = db.scalars(query.order_by(Prediction.created_at.desc()).limit(limit))
    return [prediction_payload(row) for row in rows]


@app.get(f"{settings.api_prefix}/output-catalog", tags=["predictions"])
def output_catalog() -> list[dict[str, str]]:
    """Stable, machine-readable contract for every Phase 3 output."""
    return catalog_payload()


@app.get(
    f"{settings.api_prefix}/games/{{game_id}}/predictions/{{family}}",
    tags=["predictions"],
)
def prediction_family(
    game_id: str,
    family: str,
    headline_only: bool = True,
    db: Session = Depends(get_db),
) -> list[dict]:
    if family not in {"game", "team", "pitcher", "batter"}:
        raise HTTPException(404, "Unknown prediction family")
    query = select(Prediction).where(
        Prediction.game_id == game_id,
        Prediction.category == family,
    )
    if headline_only:
        query = query.where(Prediction.is_headline.is_(True))
    rows = list(db.scalars(query.order_by(Prediction.statistic, Prediction.player_id)))
    return [prediction_payload(row) for row in rows]


@app.post(
    f"{settings.api_prefix}/predictions",
    response_model=PredictionRead,
    status_code=status.HTTP_201_CREATED,
    tags=["predictions"],
)
def post_prediction(data: PredictionCreate, db: Session = Depends(get_db)):
    prediction, created = create_revision(db, data)
    if not created:
        return prediction
    return prediction


@app.post(
    f"{settings.api_prefix}/predictions/{{prediction_id}}/resolve",
    response_model=PredictionRead,
    tags=["predictions"],
)
def resolve(
    prediction_id: str, payload: PredictionResolve, db: Session = Depends(get_db)
):
    prediction = db.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(404, "Prediction not found")
    try:
        return resolve_prediction(db, prediction, payload.result, payload.resolved_at)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post(f"{settings.api_prefix}/games/{{game_id}}/resolve", tags=["predictions"])
def resolve_game(
    game_id: str,
    payload: GameResolution,
    db: Session = Depends(get_db),
) -> dict[str, int]:
    values = payload.model_dump(exclude={"resolved_at"})
    return resolve_game_outputs(db, game_id, values, payload.resolved_at)


@app.get(f"{settings.api_prefix}/games/{{game_id}}", tags=["games"])
def game_detail(game_id: str, db: Session = Depends(get_db)) -> dict:
    rows = list(
        db.scalars(
            select(Prediction)
            .where(Prediction.game_id == game_id)
            .order_by(Prediction.created_at)
        )
    )
    if not rows:
        raise HTTPException(404, "Game not found")
    tracks = prediction_tracks(db, game_id=game_id)
    grouped = group_games([row for row in rows if row.is_headline])
    return {
        "game": grouped[0] if grouped else {"game_id": game_id},
        "timeline": [prediction_payload(row) for row in rows],
        "evaluation_tracks": {
            name: [prediction_payload(row) for row in values]
            for name, values in tracks.items()
        },
    }


@app.get(f"{settings.api_prefix}/today", tags=["slate"])
def today(
    date: str = Query(default_factory=lambda: dt.date.today().isoformat()),
    db: Session = Depends(get_db),
) -> dict:
    rows = list(
        db.scalars(
            select(Prediction).where(
                Prediction.is_headline.is_(True),
                Prediction.validity_status == "active",
                Prediction.game_id.contains(date),
            )
        )
    )
    games = group_games(rows)
    strongest = [game for game in games if game["support_score"] >= 0.6][:5]
    uncertain = [
        game
        for game in games
        if game["support_score"] < 0.6 or game["data_quality_flags"]
    ][:5]
    return {
        "date": date,
        "generated_at": dt.datetime.now(dt.UTC),
        "games": games,
        "strongest": strongest,
        "uncertain": uncertain,
        "waiting_for_lineups": [
            game for game in games if game["lineup_status"] != "confirmed"
        ],
    }


def _profile(db: Session, user: AuthUser) -> UserProfile:
    profile = db.scalar(select(UserProfile).where(UserProfile.auth_user_id == user.id))
    if not profile:
        profile = UserProfile(auth_user_id=user.id, email=user.email)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


@app.get(f"{settings.api_prefix}/profile", tags=["account"])
def get_profile(
    user: AuthUser = Depends(current_user), db: Session = Depends(get_db)
) -> dict:
    profile = _profile(db, user)
    return {
        "id": profile.id,
        "email": profile.email,
        "display_name": profile.display_name,
        "timezone": profile.timezone,
        "detail_level": profile.detail_level,
        "default_sort": profile.default_sort,
        "alert_preferences": profile.alert_preferences,
    }


@app.patch(f"{settings.api_prefix}/profile", tags=["account"])
def update_profile(
    changes: ProfileUpdate,
    user: AuthUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    profile = _profile(db, user)
    for key, value in changes.model_dump(exclude_none=True).items():
        setattr(profile, key, value)
    db.commit()
    return get_profile(user, db)


@app.get(f"{settings.api_prefix}/following", tags=["account"])
def get_following(
    user: AuthUser = Depends(current_user), db: Session = Depends(get_db)
) -> list[dict]:
    profile = _profile(db, user)
    rows = db.scalars(select(Follow).where(Follow.profile_id == profile.id))
    return [
        {
            "id": row.id,
            "entity_type": row.entity_type,
            "entity_id": row.entity_id,
            "display_name": row.display_name,
        }
        for row in rows
    ]


@app.post(f"{settings.api_prefix}/following", tags=["account"])
def add_follow(
    follow: FollowCreate,
    user: AuthUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    profile = _profile(db, user)
    existing = db.scalar(
        select(Follow).where(
            Follow.profile_id == profile.id,
            Follow.entity_type == follow.entity_type,
            Follow.entity_id == follow.entity_id,
        )
    )
    if existing:
        return {"id": existing.id, **follow.model_dump()}
    row = Follow(profile_id=profile.id, **follow.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, **follow.model_dump()}


@app.delete(f"{settings.api_prefix}/following/{{follow_id}}", status_code=204, tags=["account"])
def remove_follow(
    follow_id: str,
    user: AuthUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> None:
    profile = _profile(db, user)
    follow = db.scalar(
        select(Follow).where(Follow.id == follow_id, Follow.profile_id == profile.id)
    )
    if not follow:
        raise HTTPException(404, "Follow not found")
    db.delete(follow)
    db.commit()


@app.post(f"{settings.api_prefix}/agent/ask", response_model=AgentAnswer, tags=["agent"])
def ask(
    payload: AgentQuestion,
    user: AuthUser = Depends(current_user),
    db: Session = Depends(get_db),
) -> dict:
    return answer_question(
        db,
        payload.question,
        game_id=payload.game_id,
        detail_level=payload.detail_level,
        auth_user_id=user.id,
    )


@app.get(f"{settings.api_prefix}/model-performance", tags=["evaluation"])
def model_performance(db: Session = Depends(get_db)) -> dict:
    grouped = db.execute(
        select(
            Prediction.statistic,
            Prediction.validation_status,
            func.count(Prediction.id),
        ).group_by(Prediction.statistic, Prediction.validation_status)
    )
    return {
        "categories": [
            {"statistic": statistic, "status": status_, "n_predictions": count}
            for statistic, status_, count in grouped
        ],
        "note": "Validation labels remain provisional until frozen-data reports pass the ship gate.",
    }
