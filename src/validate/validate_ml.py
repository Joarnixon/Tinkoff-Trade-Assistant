from src.model.metrics import *

def val_ml(model, X_val, y_val, logger, monitor=None):
    y_pred, y_proba = model.predict(X_val)
    acc, prec, rec, f1_score = accuracy(y_pred, y_val), precision(y_pred, y_val), recall(y_pred, y_val), f1(y_pred, y_val)
    logger.info(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}, F1: {f1_score}')
    return (acc, prec, rec, f1_score), (X_val, y_val, y_pred, y_proba)