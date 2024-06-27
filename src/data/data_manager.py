import os
import json
import ast
from polars import read_csv, DataFrame
from typing import Union
import csv
from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig

class DataManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def _write_mean_volume_log(self, mean_volume):
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
                dataframe = DataFrame(content, schema=self.cfg.data.raw_data_columns)
            else:
                assert content.shape[1] == len(self.cfg.data.raw_data_columns), "DataFrame must have the same number of columns as raw_data_columns"
                dataframe = content
            dataframe.write_csv(f, include_header=False)

    def clear_share(self, figi: str):
        file_path = os.path.join(self.cfg.paths.raw_data, figi, f'{self.cfg.data.raw_data_filename}.csv')
        with open(file_path, 'w') as f:
            f.write(','.join(self.cfg.data.raw_data_columns) + '\n')

    def write_trend(self, content, mode='a') -> None:
        file_path = os.path.join(self.cfg.paths.trend_data, f'{self.cfg.data.raw_data_filename}.csv')
        with open(file_path, mode) as file:
            file.write(content)