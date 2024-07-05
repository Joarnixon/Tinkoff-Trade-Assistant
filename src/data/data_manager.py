import os
import json
import ast
from torch.nn import Module
import joblib
from torch import save
from polars import read_csv, DataFrame
from typing import Union, Optional
from datetime import datetime, timedelta
import hydra
from numpy import array
from omegaconf import DictConfig

class NullDataManager:
    def __init__(self, cfg: DictConfig = None) -> None:
        pass

    def write_share(self, figi: str, content: DataFrame) -> None:
        pass

    def write_processed_share(self, figi: str, content: DataFrame) -> None:
        pass

    def write_trend(self, content: DataFrame) -> None:
        pass

    def write_alerts(self, content: DataFrame) -> None:
        pass

    def write_selected_features(self, content: list[str], figi: str) -> None:
        pass

    def save_model(self, model) -> None:
        pass
        
class DataManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.share_schema = self.cfg.data.raw_data_trades_columns + [f"{column}_{proc}" for column in self.cfg.data.raw_data_orderbook_columns for proc in (self.cfg.data.data_gather.orderbook_processes + ['last']) ]

    def _write_mean_volume_log(self, mean_volume: dict):
        current_time = datetime.now()
        log_data = {
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "mean_volume": mean_volume,
            "info": {"candle": self.cfg.data.mean_volume.candle_interval,
                     "days_interval": self.cfg.data.mean_volume.update_interval_days}
        }
        with open(self.cfg.paths.mean_volume_log, "w") as log_file:
            json.dump(log_data, log_file)

    def _calc_mean_volume(self):
        print('...Calculating average volumes...')
        mean_volume = hydra.utils.call(self.cfg.data.mean_volume.gather_function, self.cfg)
        self._write_mean_volume_log(mean_volume)
        print('...Calculation complete...')
        return mean_volume

    def get_mean_volume(self):
        try:
            with open(self.cfg.paths.mean_volume_log, "r") as log_file:
                log_data = json.load(log_file)
                timestamp = datetime.strptime(log_data["timestamp"], "%Y-%m-%d %H:%M:%S")
                mean_volume = log_data["mean_volume"]
                info = log_data["info"]
                return timestamp, mean_volume, info
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None, None, None

    def read_alerts(self):
        try:
            with open(self.cfg.paths.alerts, "r") as alert_file:
                content = alert_file.read()
                return ast.literal_eval(content) if content else []
        except FileNotFoundError:
            return []

    def write_alerts(self, figis):
        with open(self.cfg.paths.alerts, "w") as alert_file:
            alert_file.write(str(figis))

    def clear_alerts(self):
        with open(self.cfg.paths.alerts, "w") as alert_file:
            alert_file.write("")

    def update_mean_volume(self):
        """
        Updates the mean volume log per share every config/update_interval_days.
        """
        last_execution_time, stored_mean_volume, info = self.get_mean_volume()

        if last_execution_time is None:
            return self._calc_mean_volume()
        
        timedelta_condition = (datetime.now() - last_execution_time) >= timedelta(days=self.cfg.data.mean_volume.update_interval_days)
        info_not_equal_condition = info["candle"] != self.cfg.data.mean_volume.candle_interval or info["days_interval"]!= self.cfg.data.mean_volume.update_interval_days
        
        if timedelta_condition or info_not_equal_condition:
            return self._calc_mean_volume()
        else:
            time_until_next_execution = timedelta(days=self.cfg.data.mean_volume.update_interval_days) - (datetime.now() - last_execution_time)
            print(f"Time until next mean volume update: {time_until_next_execution}")
            return stored_mean_volume

    def load_share(self, figi: str) -> DataFrame:
        file_path = os.path.join(self.cfg.paths.raw_data, figi, f'{self.cfg.data.raw_data_filename}.csv')
        return read_csv(file_path)

    def write_share(self, figi: str, content: Union[list, DataFrame], mode='a') -> None:
        file_path = os.path.join(self.cfg.paths.raw_data, figi, f'{self.cfg.data.raw_data_filename}.csv')
        with open(file_path, mode=mode) as f:
            if isinstance(content, list):
                assert isinstance(content[0], list), "Check the number of data dimensions"
                dataframe = DataFrame(content, schema=self.share_schema)
            else:
                assert content.shape[1] == len(self.share_schema), "DataFrame must have the same number of columns as specified in config"
                dataframe = content
            dataframe.write_csv(f, include_header=False)

    def write_processed_share(self, figi: str, content: DataFrame) -> None:
        file_path = os.path.join(self.cfg.paths.processed_data, figi, f'{self.cfg.data.processed_data_filename}.csv')
        if os.path.getsize(file_path) == 0:
            with open(file_path, mode='w') as f:
                content.write_csv(f, include_header=True)
            return

        with open(file_path, 'r') as f:
            first_line = f.readline().strip()

        file_header = first_line.split(',')
        df_columns = content.columns

        if file_header != df_columns:
            with open(file_path, mode='w') as f:
                content.write_csv(f, include_header=True)
        else:
            with open(file_path, mode='a') as f:
                content.write_csv(f, include_header=False)
        
    def write_processed_share_labels(self, figi: str, labels: array) -> None:
        labels_content = DataFrame(labels, schema=['label'])
        labels_file_path = os.path.join(self.cfg.paths.processed_data, figi, f'{self.cfg.data.processed_data_labels_filename}.csv')
    
        if os.path.getsize(labels_file_path) == 0:
            with open(labels_file_path, mode='w') as f:
                labels_content.write_csv(f, include_header=True)
            return            
        with open(labels_file_path, mode='a') as f:
            labels_content.write_csv(f, include_header=False)


    def clear_share(self, figi: str):
        file_path = os.path.join(self.cfg.paths.raw_data, figi, f'{self.cfg.data.raw_data_filename}.csv')
        raw_data_orderbook_columns = [f"{column}_{proc}" for column in self.cfg.data.raw_data_orderbook_columns for proc in (self.cfg.data.data_gather.orderbook_processes + ['last'])]
        raw_data_columns = self.cfg.data.raw_data_trades_columns + raw_data_orderbook_columns
        with open(file_path, 'w') as f:
            f.write(','.join(raw_data_columns) + '\n')

    def write_trend(self, content: DataFrame) -> None:
        file_exists = os.path.isfile(self.cfg.paths.trend_data)
        
        if not file_exists:
            with open(self.cfg.paths.trend_data, mode='w') as f:
                f.write(','.join(self.cfg.data.trend_data_columns) + '\n')

        with open(self.cfg.paths.trend_data, mode='a') as f:
            dataframe = DataFrame(content, schema=self.cfg.data.trend_data_columns)
            dataframe.write_csv(f, include_header=False)
    
    def write_selected_features(self, content: list[str], figi: Optional[str]) -> None:
        file_exists = os.path.isfile(self.cfg.paths.selected_features)

        if not file_exists:
            with open(self.cfg.paths.selected_features, mode='w') as f:
                json.dump({figi: content}, f)
        else:
            with open(self.cfg.paths.selected_features, mode='r') as f:
                selected_features_dict = json.load(f)

            selected_features_dict[figi] = content

            with open(self.cfg.paths.selected_features, mode='w') as f:
                json.dump(selected_features_dict, f)
    
    def save_model(self, model, figi, save_best=False):
        model_type = model.__class__.__name__
        model_path = os.path.join(self.cfg.paths.models, figi)

        if isinstance(model, Module):
            # Save PyTorch model
            checkpoint = {'model': model, 'state_dict': model.state_dict()}
            if not save_best:
                save(checkpoint, f"{model_path}/{model_type}.pth")
            else:
                save(checkpoint, f"{model_path}/best.pth")
        else:
            if not save_best:
                joblib.dump(model, f"{model_path}/{model_type}.joblib")
            else:
                joblib.dump(model, f"{model_path}/best.joblib")