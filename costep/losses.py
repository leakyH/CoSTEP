import torch


def build_elementwise_loss_tensor(prediction, target, loss_name):
    if loss_name != "MAE":
        raise ValueError("Daily CoSTEP uses MAE; interval CoSTEP-IR uses reportMAE")
    return torch.abs(prediction - target)
