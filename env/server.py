from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from .environment import SmartEmailTriageEnv
from .models import Action

app = FastAPI(title="Smart Email Triage OpenEnv", version="1.0.0")
env = SmartEmailTriageEnv(task_mode="hard", seed=42)


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "smart-email-triage-openenv"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global env
    payload = payload or {}
    task_mode = payload.get("task_mode", "hard")
    seed = int(payload.get("seed", 42))
    max_steps = int(payload.get("max_steps", 30))

    if task_mode not in {"easy", "medium", "hard"}:
        raise HTTPException(status_code=400, detail="task_mode must be one of easy|medium|hard")

    env = SmartEmailTriageEnv(task_mode=task_mode, seed=seed, max_steps=max_steps)
    obs = env.reset()
    return {"observation": obs.model_dump(), "done": False}


@app.post("/openenv/reset")
def openenv_reset(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return reset(payload)


@app.post("/step")
def step(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        action = Action(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {exc}") from exc

    if action.action_type == "classify_email" and action.classification is None:
        raise HTTPException(status_code=400, detail="classification is required for classify_email")
    if action.action_type == "reply_email" and not action.reply_text:
        raise HTTPException(status_code=400, detail="reply_text is required for reply_email")
    if action.action_type == "prioritize_email" and action.priority_rank is None:
        raise HTTPException(status_code=400, detail="priority_rank is required for prioritize_email")

    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward.model_dump(),
        "done": result.done,
        "info": result.info,
    }


@app.post("/openenv/step")
def openenv_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    return step(payload)


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state()


@app.get("/openenv/state")
def openenv_state() -> Dict[str, Any]:
    return state()
