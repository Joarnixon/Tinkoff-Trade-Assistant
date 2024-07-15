import asyncio
from tinkoff.invest import AsyncClient
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import now
from tinkoff.invest.utils import quotation_to_decimal as qtd
from datetime import datetime, timedelta
from src.data.data_manager import DataManager
from src.data.observer import Observer
from src.data.subject import Subject
import hydra
from collections import deque
from itertools import islice
import logging
from omegaconf import DictConfig

class VolumeScan(Subject):
    def __init__(self, cfg, data_manager: DataManager, log: logging.Logger=None):
        super().__init__()
        self.cfg = cfg
        self.data_manager = data_manager
        self.log = log if log is not None else logging.getLogger("tinkoff.invest.logging")
        self.log.setLevel(logging.WARNING)
        self.price_change_threshold = float(cfg.volume_scan.price_change_threshold)
        self.bigger_by_factor = float(cfg.volume_scan.bigger_by_factor)

        sessions_config = {"1_MIN": [2, CandleInterval.CANDLE_INTERVAL_1_MIN], "2_MIN": [3, CandleInterval.CANDLE_INTERVAL_2_MIN], "3_MIN": [4, CandleInterval.CANDLE_INTERVAL_3_MIN], "5_MIN": [6, CandleInterval.CANDLE_INTERVAL_5_MIN], "10_MIN": [11, CandleInterval.CANDLE_INTERVAL_10_MIN], "15_MIN": [16, CandleInterval.CANDLE_INTERVAL_15_MIN]}
        self.session_time = sessions_config[cfg.volume_scan.session_time]
        self.time_delay = (self.session_time[0] - 1) * 60
        timestamp, self.mv, info = self.data_manager.get_mean_volume()
    
        self.buffer = {figi: deque([], maxlen=120) for figi in self.mv}


    async def process(self):
        delay = self.time_delay
        session = 0
        while True:
            await asyncio.sleep(delay)
            await self.process_data_points()
            session += 1
            print(f'Session {session} done')

    async def run(self):
        await self.process()

    async def process_data_points(self):
        try:
            async with AsyncClient(self.cfg.tokens.TOKEN) as client:
                factor = self.bigger_by_factor
                threshold = self.price_change_threshold
                session_time = self.session_time
                for figi in self.mv:
                    if self.mv[figi] == 0:
                        self.log.info(f"Mean volume for {figi} is not available.")
                        continue
                    candle_data = await self.get_candle_data(client, figi, self.mv[figi], threshold, factor, session_time)
                    if candle_data is not None:
                        print(f'{figi} : {candle_data[0]} %, volume bigger than average {self.cfg.data.mean_volume.candle_interval} candle volume by: {candle_data[1]/self.mv[figi]} times!')
        except Exception as e:
            self.log.exception(e)
            pass

    async def get_candle_data(self, client, figi, mv, threshold, factor, session_time):
        async for candles in client.get_all_candles(
            figi=figi,
            from_=now() - timedelta(minutes=session_time[0]),
            interval=session_time[1],
        ):
            candle = candles
            self.buffer[figi].append(candle)
            price_change = float(round(100 * (qtd(candle.close) - qtd(candle.open)) / qtd(candle.open), 3))
            volume = candle.volume
            if abs(price_change) > threshold and volume > factor * mv:
                self.notify({figi: [price_change, volume]})
                return [price_change, volume]
            return None
        
    async def get_candles_from_buffer(self, figi: str, info: int):
        if figi not in self.buffer:
            return []
        l = len(self.buffer[figi]) 
        num_candles_to_return = max(l - info, 0)
        candles = list(islice(self.buffer[figi], info, l))
        return candles
    
# async def run_vs(cfg: DictConfig) -> None:
#     dm = DataManager(cfg)
#     vs = VolumeScan(cfg, dm)
#     await vs.run()

# @hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
# def main(cfg: DictConfig) -> None:
#     asyncio.run(run_vs(cfg))

# log = logging.getLogger("tinkoff.invest.logging")
# log.setLevel(logging.WARNING)
# main()