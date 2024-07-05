import torch
from numpy import argmax
from src.model.metrics import *

def val_nn(model, val_loader, criterion, device, return_data=False):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device).squeeze(-1)
            output, preds = model(data)
            loss = criterion(output, label)
            val_loss += loss.item()
            all_preds.extend(torch.softmax(preds, dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    predicted_labels = argmax(all_preds, axis=1)
    # Calculate metrics
    accuracy_score = accuracy(all_labels, predicted_labels)
    precision_score = precision(all_labels, predicted_labels)
    recall_score = recall(all_labels, predicted_labels)
    f1_score = f1(all_labels, predicted_labels)
    
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score
    }
    if return_data:
        return val_loss, metrics, all_preds
    
    return val_loss, metrics