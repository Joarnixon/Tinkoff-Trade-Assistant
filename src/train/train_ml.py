import time
from sklearn.model_selection import train_test_split

def train_ml(model, data, labels, validate_func, logger, monitor, cfg):

    data_train, data_val, labels_train, labels_val = train_test_split(
        data, labels, 
        test_size=cfg.test_size, 
        random_state=cfg.random_state, 
        shuffle=cfg.shuffle, 
        stratify=labels if cfg.stratify else None)
    
    start_time = int(time.time())
    model.fit(data_train, labels_train)
    logger.info(f'Model trained in {int(time.time()) - start_time} seconds')

    validation_metrics, validation_data = validate_func(model, data_val, labels_val, logger)

    return model, None, validation_metrics, validation_data
