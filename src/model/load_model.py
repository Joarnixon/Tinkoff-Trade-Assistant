import torch

def load_nn(filename):
    checkpoint = torch.load(filename)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = True
    
    model.eval()
    
    return model