import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.model.loss import cross_entropy_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from polars import DataFrame
from copy import deepcopy
from numpy import array, argmax

def train_nn(model: nn.Module, dataset: DataFrame, labels: array, validate_func: callable, logger, monitor: SummaryWriter, cfg):
    device = torch.device(cfg.device)
    
    data_train, data_val, labels_train, labels_val = train_test_split(
        dataset, labels, 
        test_size=cfg.test_size, 
        random_state=cfg.random_state, 
        shuffle=cfg.shuffle, 
        stratify=labels if cfg.stratify else None
    )
    torch.manual_seed(cfg.random_state)
    train_dataset = TensorDataset(torch.tensor(data_train.drop('time').to_numpy(), dtype=torch.float32), 
                                  torch.tensor(labels_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(data_val.drop('time').to_numpy(), dtype=torch.float32), 
                                torch.tensor(labels_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, drop_last=True)
    
    model = model.to(device)
    optimizer = getattr(torch.optim, cfg.optimizer.name)(model.parameters(), **cfg.optimizer.params)
    criterion = cross_entropy_loss
    best_model = deepcopy(model)
    best_val_loss = float('inf')
    best_val_score = None
    early_stopping_counter = 0
    
    if cfg.weights == 'balanced':
        label_counts = torch.bincount(torch.tensor(labels, dtype=int))
        total_samples = len(labels)
        class_weights = total_samples / (label_counts * len(label_counts))
    elif cfg.weights is None:
        class_weights = torch.tensor([1, 1, 1])
    else:
        class_weights = torch.tensor(cfg.weights)

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = 0.0
        
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze(-1)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, label, class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        val_loss, val_metrics = validate_func(model, val_loader, criterion, device, class_weights)
        
        logger.info(f"Epoch {epoch+1}/{cfg.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if monitor:
            monitor.add_scalar('Train loss', train_loss, epoch+1)
            monitor.add_scalar('Val loss', val_loss, epoch+1)
            monitor.add_scalar('Accuracy', val_metrics['accuracy'], epoch+1)  

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_score = val_metrics['f1_score']
            early_stopping_counter = 0
            best_model = deepcopy(model)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
    hparams = {'batch_size': cfg.batch_size,
            'lr': cfg.optimizer.params.lr,
            'weight_decay': cfg.optimizer.params.weight_decay,
            'stratification': cfg.stratify,
            'hidden_layers': cfg.hidden_layers,
            'select_features': cfg.select_features,
            'percent_change_threshold': cfg.percent_change_threshold,
            'min_time_interval': cfg.min_time_interval}
    metrics = {'Accuracy': best_val_score}

    monitor.add_hparams(hparam_dict=hparams, metric_dict=metrics, run_name='hparam')
    
    val_loss, validation_metrics, probas = validate_func(best_model, val_loader, criterion, device, class_weights, return_data=True)

    return best_model, best_val_score, validation_metrics, (data_val, labels_val, argmax(probas, axis=1), probas)


