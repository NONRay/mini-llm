import matplotlib.pyplot as plt



def show_attention(att):

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
