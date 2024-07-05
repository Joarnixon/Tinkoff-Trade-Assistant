import hydra
from src.data.data_gather import DataCollector
from src.pipelines.online_pipeline import OnlinePredictions
from src.visualise import OnlinePlot
from src.data.data_manager import DataManager
import logging
import asyncio

class Observer:
    def update(self, data: dict):
        pass

test_observer = Observer()


logger = logging.getLogger("tinkoff.invest.logging")
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
def main(cfg):
    figi = 'TCS00A106YF0'
    dm = DataManager(cfg)
    dc = DataCollector(cfg)

    # specific figi to make predictions for or don't pass
    op = OnlinePredictions(cfg, figi)
    
    dc.attach(op)
    op.attach(test_observer)

    async def run_tasks():
        await asyncio.gather(
            dc.run(),
            op.run()
        )

    asyncio.run(run_tasks())

if __name__ == "__main__":
    main()