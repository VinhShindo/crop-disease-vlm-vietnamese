import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def plot_tsne(

    embeddings,
    labels,

    save_path
):

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    reduced = tsne.fit_transform(
        embeddings
    )

    df = pd.DataFrame({

        "x": reduced[:, 0],

        "y": reduced[:, 1],

        "label": labels
    })

    plt.figure(
        figsize=(10, 8)
    )

    sns.scatterplot(

        data=df,

        x="x",
        y="y",

        hue="label"
    )

    plt.title(
        "t-SNE Embedding Visualization"
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()