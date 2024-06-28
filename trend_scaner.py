import asyncio
from datetime import datetime
from src.data.observer import Observer
import hydra
from omegaconf import DictConfig
from src.data.data_manager import DataManager, NullDataManager
from src.data.data_gather import DataCollector

class TrendScan(Observer):
    def __init__(self, data_manager=None, time_delay=60):
        super().__init__()
        self.data_points = []
        self.session = 0
        self.time_delay = time_delay
        self.data_manager = data_manager if data_manager is not None else NullDataManager()

    async def process(self):
        delay = self.time_delay
        while True:
            await asyncio.sleep(delay)
            self.process_data_points()
            self.session += 1

    async def run(self):
        await self.process()

    def update(self, data_point: dict):
        self.data_points.append(list(data_point.values())[0])

    def process_data_points(self):
        plus1q, minus1q, plus1, minus1 = 0, 0, 0, 0
        for point in list(self.data_points):
            plus1 += point[0]
            plus1q += point[1]
            minus1 += point[2]
            minus1q += point[3]
        self.data_points = []

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_point = [[timestamp, plus1, plus1q, minus1, minus1q]]
        self.data_manager.write_trend(data_point)
        
        if any([plus1q == 0, minus1q == 0, plus1 == 0, minus1 == 0]):
            print('Недопустимые данные. Торги вероятно прекращены')
            return

        if plus1q > minus1q:
            print(f'Сессия {self.session}, тренд РОСТ. ОБЪЕМ ПОКУПАТЕЛЕЙ БОЛЬШЕ в {plus1q/minus1q} раз')
            if plus1 < minus1:
                print(f'\t В тоже время, КОЛ-ВО ПРОДАВЦОВ БОЛЬШЕ в {minus1/plus1} раз')
        else:
            print(f'Сессия {self.session}, тренд ПАДЕНИЕ. ОБЪЕМ ПРОДАВЦОВ БОЛЬШЕ в {minus1q/plus1q} раз')

            if plus1 > minus1:
                print(f'\t В тоже время, КОЛ-ВО ПОКУПАТЕЛЕЙ БОЛЬШЕ в {plus1/minus1} раз')

async def run_dc_and_ts(cfg: DictConfig) -> None:
    dc = DataCollector(cfg)
    dm = DataManager(cfg)

    ts = TrendScan(dm, time_delay=cfg.trend_scan.time_delay)
    dc.attach(ts)
    await asyncio.gather(dc.run(), ts.run())

@hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
def main(cfg: DictConfig) -> None:
    asyncio.run(run_dc_and_ts(cfg))

main()