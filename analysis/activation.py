import torch


def extract_hidden(
    model,
    tokens
):

    model.eval()


    with torch.no_grad():

        logits,loss,hidden=model(
            tokens,
            return_hidden=True
        )


    return hidden

