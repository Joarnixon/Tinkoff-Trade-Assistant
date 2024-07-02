import time

def train_ml(model, X_train, y_train, logger, monitor):
    start_time = int(time.time())
    model.fit(X_train, y_train)
    logger.info(f'Model trained in {int(time.time()) - start_time} seconds')
    
    return model, None
