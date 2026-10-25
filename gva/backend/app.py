"""Localhost API for the Pi. The ops dashboard reads Supabase, not this port."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request

from gva import __version__
from gva.backend.sync_supabase import _write_setting
from gva.config import ConfigChangeError, env_value, load_config, validate_change
from gva.ids import utc_now_ms
from gva.runtime.bootstrap import prepare_dirs
from gva.telemetry.store import Store

ROLES = {"viewer": 1, "reviewer": 2, "developer": 3, "owner": 4}


def create_app(cfg=None, store: Store | None = None) -> FastAPI:
    cfg = prepare_dirs(cfg or load_config())
    store = store or Store(Path(cfg.telemetry.db_path))
    docs = bool(cfg.backend.docs_enabled and cfg.device.env != "prod")
    app = FastAPI(
        title="GVA",
        version=__version__,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    def require(minimum: str):
        def dependency(request: Request) -> dict[str, Any]:
            secret = env_value("GVA_API_TOKEN_SECRET")
            if not secret:
                raise HTTPException(status_code=503, detail="GVA_API_TOKEN_SECRET is not set")
            header = request.headers.get("authorization", "")
            if not header.lower().startswith("bearer "):
                raise HTTPException(status_code=401, detail="missing token")
            token = header.split(" ", 1)[1].strip()
            try:
                payload = jwt.decode(token, secret, algorithms=["HS256"])
            except jwt.PyJWTError as exc:
                raise HTTPException(status_code=401, detail="invalid token") from exc
            role = str(payload.get("role") or "")
            if ROLES.get(role, 0) < ROLES[minimum]:
                raise HTTPException(status_code=403, detail="forbidden")
            return payload

        return dependency

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "device_id": cfg.device.id, "version": __version__}

    @app.get("/v1/turns")
    def turns(_user: dict = Depends(require("viewer"))) -> dict[str, Any]:
        return {"turns": store.recent("turn.completed", 50)}

    @app.post("/v1/turns/{turn_id}/flag")
    def flag(turn_id: str, _user: dict = Depends(require("reviewer"))) -> dict[str, Any]:
        store.flag_turn(turn_id, utc_now_ms())
        return {"flagged": turn_id}

    @app.get("/v1/usage")
    def usage(_user: dict = Depends(require("viewer"))) -> dict[str, Any]:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {"spent_usd_today": round(store.spend_since(day + "T00:00:00.000Z"), 6)}

    @app.get("/v1/config")
    def show_config(_user: dict = Depends(require("developer"))) -> dict[str, Any]:
        return cfg.model_dump(mode="json")

    @app.post("/v1/config")
    def set_config(body: dict[str, Any], _user: dict = Depends(require("owner"))) -> dict[str, Any]:
        key = str(body.get("key") or "")
        try:
            value = validate_change(key, body.get("value"))
        except ConfigChangeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if key != "action.health_check":
            _write_setting(cfg, key, value)
        return {"ok": True, "key": key}

    return app


app = create_app()
