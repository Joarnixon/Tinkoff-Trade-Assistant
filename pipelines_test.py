from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate
from src.pipelines.data_pipeline import DataPipelineFactory
from src.pipelines.train_pipeline import TrainPipelineFactory
from src.data.data_manager import DataManager
from config import update_processed_data_folder
import logging



logger = logging.getLogger("tinkoff.invest.logging")
logger.setLevel(logging.INFO)



@hydra.main(version_base=None, config_path='D:/Tinkoff-Trade-Assistant/config', config_name='general.yaml')
def main(cfg: DictConfig) -> None:
    update_processed_data_folder(cfg)
    figi = 'TCS00A106YF0'
    dm = DataManager(cfg)
    sh = dm.load_share(figi)
    model_cfg = OmegaConf.load('config/models/logistic_regression.yaml')
    model = instantiate(model_cfg)
    dp = DataPipelineFactory.create_offline_pipeline(cfg)
    tp = TrainPipelineFactory.create_ml_pipeline(cfg=cfg, model=model, logger=logger, data_manager=dm)
    sh = dp.transform(sh)
    model, _, _ = tp.train(sh, figi)

main()