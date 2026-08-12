import torch


def extract_hidden(
    model,
    tokens
):
    """提取每个 Transformer 层的隐藏状态，用于可视化或分析。"""

    model.eval()


    with torch.no_grad():

        logits,loss,hidden=model(
            tokens,
            return_hidden=True
        )


    return hidden
