from omegaconf import OmegaConf, DictConfig
import hydra
from src.data.data_gather import DataCollector
from src.data.data_manager import DataManager
import config
import asyncio
import yaml
import polars as pl
import logging

class Observer:
    def update(self, data: dict):
        print("Data recieved")
        print(data)

logging.getLogger("tinkoff.invest.logging").setLevel(logging.INFO)

config.update_files()
test_observer = Observer()

@hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
def main(cfg: DictConfig) -> None:
    dc = DataCollector(cfg)
    dc.attach(test_observer)
    asyncio.run(dc.run())

main()