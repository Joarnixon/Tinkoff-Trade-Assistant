from omegaconf import OmegaConf
from fastapi import APIRouter, Depends, Request
from functools import cache
import os

router = APIRouter(prefix="/shares")

def model_available(figi: str, model_path) -> bool:
    model_file = os.path.join(model_path, figi, "best.pth")
    return True

@cache
def get_trading_shares(request: Request):
    shares_path = str(request.app.state.cfg.paths.shares_dict)
    all_shares = OmegaConf.load(shares_path)
    model_path = str(request.app.state.cfg.paths.models)
    available_shares = {figi: data for figi, data in all_shares.items() if model_available(figi, model_path)}
    return available_shares

@router.get('/')
def get_shares_endpoint(shares = Depends(get_trading_shares)):
    return shares