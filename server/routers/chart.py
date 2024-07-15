from fastapi import APIRouter, Depends, HTTPException, Request
from cachetools import cached, TTLCache
from tinkoff.invest.utils import now
from tinkoff.invest import AsyncClient
from tinkoff.invest import CandleInterval
from datetime import timedelta
import logging

cache = TTLCache(maxsize=128, ttl=60)  
router = APIRouter(prefix="/chart")

logger = logging.getLogger(__name__)

@router.get('/')
def start_page():
    return None

def get_volume_scanner(request: Request):
    return request.app.state.volume_scanner

@router.get("/{figi}/{info}")
async def get_candles_data_endpoint(request: Request, figi: str, info: int, vs = Depends(get_volume_scanner)):
    try:
        data = await vs.get_candles_from_buffer(figi, info)
        return data
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))