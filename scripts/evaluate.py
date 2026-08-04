import torch


def evaluate(model, loader, device=None):

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    losses = []

    with torch.no_grad():

        for x, y in loader:

            _, loss = model(
                x.to(device),
                y.to(device),
            )

            losses.append(loss.item())

    return sum(losses) / len(losses)
