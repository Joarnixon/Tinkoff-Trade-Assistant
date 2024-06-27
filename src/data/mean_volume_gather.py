from omegaconf import DictConfig
import yaml
import json
import logging
import threading
import sys
import time
from queue import Queue
from datetime import timedelta, datetime
from tinkoff.invest import CandleInterval, Client, RequestError
from tinkoff.invest.utils import now
log = logging.getLogger(__name__)
logging.getLogger(__name__).setLevel(logging.WARNING)

def get_candle_data(figi, token, candle_interval, interval, result_queue):
    """
    This function retrieves candle data for a specific FIGI,
    calculates the average volume for given candle interval on a range of 'interval' days.

    Parameters:
    figi (str): The FIGI of the stock.
    token (str): The API token for Tinkoff Invest API.
    candle_interval (str): The interval of candles
    interval (int): The number of days to retrieve candle data for.
    result_queue (Queue): The queue to put the result (FIGI, average volume) into.
    """
    volume_week = 0
    candles_amount = 0
    candle_intervals = {'5_MIN': CandleInterval.CANDLE_INTERVAL_5_MIN,
                        '10_MIN': CandleInterval.CANDLE_INTERVAL_10_MIN,
                        '15_MIN': CandleInterval.CANDLE_INTERVAL_15_MIN,
                        '30_MIN': CandleInterval.CANDLE_INTERVAL_30_MIN,
                         '1_HOUR': CandleInterval.CANDLE_INTERVAL_HOUR}
    candle_interval = candle_intervals[candle_interval]
    with Client(token) as client:
        try:
            for candle in client.get_all_candles(
                    figi=figi,
                    from_=now() - timedelta(days=interval),
                    interval=candle_interval
            ):  
                candles_amount += 1
                volume_week += candle.volume
        except RequestError:
            log.info('RequestError at', figi)

    if candles_amount > 0:
        result_queue.put((figi, volume_week / candles_amount))
    else:
        log.warn(f"Warning: Average volume for FIGI '{figi}' is 0. Try setting lower api_wait_time in config.yaml")
        result_queue.put((figi, 0))

# @hydra.main(version_base=None, config_path=f'{os.getcwd()}\config', config_name='general.yaml')
def get_mean_volumes(cfg: DictConfig) -> dict:
    """
    This function retrieves average volumes for a list of stocks (specified in a YAML file)
    and saves the results to a JSON file.
    """
    threads = []
    result_queue = Queue()
    with open(cfg.paths.shares_dict, 'r', encoding='utf-8') as file:
        stocks = list(yaml.safe_load(file).keys()) # only figi needed

    for i, figi in enumerate(stocks):
        thread = threading.Thread(target=get_candle_data, args=(figi,
                                                                cfg.tokens.TOKEN,
                                                                cfg.data.mean_volume.candle_interval,
                                                                cfg.data.mean_volume.mean_interval,
                                                                result_queue))
        threads.append(thread)
        thread.start()
        
        time.sleep(cfg.data.mean_volume.api_wait_time)
        
        # Update progress bar
        sys.stdout.write(('=' * (i + 1) + (' ' * (len(stocks) - i - 1)) + ("\r [ %d" % ((i + 1) / len(stocks) * 100) + "% ] ")))
        sys.stdout.flush()

    for thread in threads:
        thread.join()

    sys.stdout.write('\n')  # Move to the next line after progress bar

    volume_data = {}
    while not result_queue.empty():
        figi, avg_volume = result_queue.get()
        volume_data[figi] = avg_volume
        
    return volume_data