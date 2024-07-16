from omegaconf import OmegaConf, DictConfig
import hydra
from src.data.data_gather import DataCollector
from src.data.data_manager import DataManager
import config
import asyncio
import logging

logging.getLogger("tinkoff.invest.logging").setLevel(logging.INFO)

## config.update_files()
@hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
def main(cfg: DictConfig) -> None:
    dm = DataManager(cfg)
    dc = DataCollector(cfg, data_manager=dm)
    asyncio.run(dc.run())

main()