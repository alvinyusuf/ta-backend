import torch

def bitwise_accuracy(logits_str, target_str):
    logits = [int(bit) for bit in logits_str]
    target = [int(bit) for bit in target_str]

    preds = (torch.tensor(logits) > 0).float()
    acc = 1.0 - torch.mean(torch.abs(preds - torch.tensor(target)))
    return f"{acc.item() * 100:.2f}%"
    # return str(acc.item())