from omegaconf import OmegaConf, DictConfig
import hydra
from src.data.data_manager import DataManager
import config
import yaml
import polars as pl
import logging

logging.getLogger("tinkoff.invest.logging").setLevel(logging.WARNING)

config.update_files()

# @hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
# def main(cfg: DictConfig) -> None:
#     dm = DataManager(cfg)   
#     data_point = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#     dm.write_share('BBG0100R9963', data_point)
#     sh = dm.load_share('BBG0100R9963')
#     print(sh)
# main()