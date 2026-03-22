import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from replay_repository import ReplayRepository

app = FastAPI(title="Traitors AI Replay Server")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(
    os.getenv(
        "TRAITORS_RESULTS_DIR",
        Path(__file__).parent.parent / "results" / "test_runs",
    )
)
RUN_FILTER = os.getenv("TRAITORS_RUN_FILTER", "run_final_2")
ALLOWED_RUN_IDS = [run_id.strip() for run_id in RUN_FILTER.split(",") if run_id.strip()]

repository = ReplayRepository(RESULTS_DIR, allowed_run_ids=ALLOWED_RUN_IDS)


@app.get("/")
def read_root():
    return {
        "message": "Traitors AI Replay Server",
        "version": "1.1.0",
        "results_dir": str(RESULTS_DIR),
        "allowed_run_ids": ALLOWED_RUN_IDS,
    }


@app.get("/games")
def list_games():
    return repository.list_games()


@app.get("/games/{game_id}/summary")
def get_game_summary(game_id: str):
    try:
        return repository.get_summary(game_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/games/{game_id}/events")
def get_game_events(game_id: str):
    try:
        return repository.get_events(game_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/games/{game_id}/personas")
def get_game_personas(game_id: str):
    try:
        return repository.get_personas(game_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
