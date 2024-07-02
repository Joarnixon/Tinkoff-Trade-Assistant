from tinkoff.invest import Client, Share
from typing import Optional
import hydra
from omegaconf import DictConfig
import yaml
import os


def available_share(share: Share) -> Optional[tuple]:
    '''
    Limitations for a low-risk and affordable trading.
    '''
    share_is_common = share.share_type == 1
    share_is_ru_currency = share.currency == 'rub'
    share_is_api_available = share.api_trade_available_flag is True
    share_is_not_for_qual = share.for_qual_investor_flag is False
    share_is_moex_spb = share.real_exchange == 1 or share.real_exchange == 2
    
    condition = share_is_common and share_is_ru_currency and share_is_api_available and share_is_not_for_qual and share_is_moex_spb
    return (share.figi, share.name) if condition else None

def get_all_shares(token) -> dict:
    """
    This function retrieves all available shares from the Tinkoff Invest API.
    It filters the shares based on certain conditions and returns a dictionary containing the FIGI codes and names of the available shares.

    Parameters:
    None

    Returns:
    dict: A dictionary where the keys are FIGI codes and the values are share names.
    """
    share_info = {}
    with Client(token) as client:
        # Fetch all shares from the API
        shares = client.instruments.shares().instruments
        for share in shares:
            # Check if the share meets the specified conditions
            if share_checked:= available_share(share):
                # If the share meets the conditions, add it to the dictionary
                share_info[share_checked[0]] = share_checked[1]
    return share_info

@hydra.main(version_base=None, config_path='../config', config_name='general.yaml')
def update_files(cfg: DictConfig) -> None:
    '''
    Retrieves all available shares from the Tinkoff Invest API and updates the shares.yaml file.
    Creates new folders for storing the shares backlog files. All stored data will be reset.
    '''
    update_shares_yaml(cfg)
    update_raw_data_folder(cfg)
    update_processed_data_folder(cfg)

def update_shares_yaml(cfg: DictConfig) -> None:
    """
    This function updates the shares.yaml file with the latest available shares from the Tinkoff Invest API.
    """
    with open(cfg.paths.shares_dict, 'w', encoding="utf-8") as file:
        yaml.safe_dump(get_all_shares(cfg.tokens.TOKEN), file, allow_unicode=True, default_flow_style=False)

def update_raw_data_folder(cfg: DictConfig) -> None:
    """
    This function updates the backlog of shares by creating empty folders for each share in the shares.yaml file
    and removing folders for delisted shares.
    """
    with open(cfg.paths.shares_dict, 'r', encoding="utf-8") as file:
        shares_dict = yaml.safe_load(file)

    # create raw data folder
    os.makedirs(cfg.paths.raw_data, exist_ok=True)
    existing_folders = set(os.listdir(cfg.paths.raw_data))
    valid_figis = set(shares_dict.keys())
    delisted_figis = existing_folders.difference(valid_figis)

    for figi in shares_dict:
        # create empty folder
        folder_path = os.path.join(cfg.paths.raw_data, figi)
        os.makedirs(folder_path, exist_ok=True)
        # create csv
        
        file_path = os.path.join(folder_path, f'{cfg.data.raw_data_filename}.csv')
        raw_data_orderbook_columns = [f"{column}_{proc}" for column in cfg.data.raw_data_orderbook_columns for proc in (cfg.data.data_gather.orderbook_processes + ['last'])]
        raw_data_columns = cfg.data.raw_data_trades_columns + raw_data_orderbook_columns
        with open(file_path, 'w') as f:
            f.write(','.join(raw_data_columns) + '\n')
    # remove any delisted share
    for figi in delisted_figis:
        folder_path = os.path.join(cfg.paths.raw_data, figi)
        os.rmdir(folder_path)

def update_processed_data_folder(cfg: DictConfig) -> None:
    with open(cfg.paths.shares_dict, 'r', encoding="utf-8") as file:
        shares_dict = yaml.safe_load(file)

    os.makedirs(cfg.paths.processed_data, exist_ok=True)
    existing_folders = set(os.listdir(cfg.paths.processed_data))
    valid_figis = set(shares_dict.keys())
    delisted_figis = existing_folders.difference(valid_figis)

    for figi in shares_dict:
        folder_path = os.path.join(cfg.paths.processed_data, figi)
        os.makedirs(folder_path, exist_ok=True)
        
        file_path = os.path.join(folder_path, f'{cfg.data.processed_data_filename}.csv')
        with open(file_path, 'w') as f:
            pass
        labels_file_path = os.path.join(folder_path, f'{cfg.data.processed_data_labels_filename}.csv')
        with open(labels_file_path, 'w') as f:
            pass

    # remove any delisted share
    for figi in delisted_figis:
        folder_path = os.path.join(cfg.paths.processed_data, figi)
        os.rmdir(folder_path)



    
    
    

