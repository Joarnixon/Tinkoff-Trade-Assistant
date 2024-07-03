import torch
from src.model.metrics import *

def val_nn(model, val_loader, criterion, device):
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
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    accuracy_score = accuracy(all_labels, all_preds)
    precision_score = precision(all_labels, all_preds)
    recall_score = recall(all_labels, all_preds)
    f1_score = f1(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score
    }
    return val_loss, metrics