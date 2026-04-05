from __future__ import annotations

from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import load_config
from .models import build_model
from .preprocess import build_features


class PredictRequest(BaseModel):
    psr: List[List[float]] = Field(..., description="[T,S] pseudo-range matrix")
    presence: List[List[float]] = Field(..., description="[T,S] visibility mask")


class PredictResponse(BaseModel):
    spoof_probability: List[float]


app = FastAPI(title="ANIMA GNSS Spoofing Detector")

_cfg = load_config(None)
_device = torch.device("cpu")
_model = build_model(_cfg).to(_device)
_model.eval()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    psr = torch.tensor(req.psr, dtype=torch.float32, device=_device).unsqueeze(0)
    presence = torch.tensor(req.presence, dtype=torch.float32, device=_device).unsqueeze(0)
    features = build_features(psr, presence)

    with torch.no_grad():
        logits = _model(features, presence.bool())
        probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().tolist()

    return PredictResponse(spoof_probability=probs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("anima_deep_seq2seq_gnss.api:app", host="0.0.0.0", port=8080, reload=False)
