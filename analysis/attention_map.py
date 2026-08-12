import matplotlib.pyplot as plt



def show_attention(att):
    """绘制单个注意力矩阵（行是查询位置，列是键位置）。"""

    plt.imshow(
        att
    )

    plt.xlabel(
        "Key token"
    )

    plt.ylabel(
        "Query token"
    )

    plt.show()
