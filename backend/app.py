from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

app = FastAPI(title="Traitors AI Replay Server")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to results directory
RESULTS_DIR = Path(__file__).parent.parent / "results" / "logs"


@app.get("/")
def read_root():
    return {"message": "Traitors AI Replay Server", "version": "1.0.0"}


@app.get("/games")
def list_games() -> List[Dict[str, Any]]:
    """List all available game logs."""
    if not RESULTS_DIR.exists():
        return []
    
    games = []
    for file in RESULTS_DIR.glob("*_summary.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                summary = json.load(f)
                games.append(summary)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Sort by game_id (most recent first)
    games.sort(key=lambda x: x.get("game_id", ""), reverse=True)
    return games


@app.get("/games/{game_id}/summary")
def get_game_summary(game_id: str) -> Dict[str, Any]:
    """Get summary for a specific game."""
    summary_path = RESULTS_DIR / f"{game_id}_summary.json"
    
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/games/{game_id}/events")
def get_game_events(game_id: str) -> List[Dict[str, Any]]:
    """Get all events for a specific game from JSONL log."""
    log_path = RESULTS_DIR / f"{game_id}.jsonl"
    
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Game log {game_id} not found")
    
    events = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    
    return events


@app.get("/games/{game_id}/personas")
def get_game_personas(game_id: str) -> Dict[int, Dict[str, Any]]:
    """Extract persona assignments from game events."""
    events = get_game_events(game_id)
    personas = {}
    
    # Find persona assignment events
    for event in events:
        if event.get("action_type") == "assign_persona":
            player_id = event.get("actor_id")
            personas[player_id] = event.get("payload", {}).get("persona", {})
    
    return personas


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
