import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("sbert_models.csv")
df = df.drop_duplicates().sort_values("AMI")

plt.rcParams.update({"font.size": 13, "font.family": "serif"})

metrics = ("ARI", "AMI")
fr_data = df[df.lang == "fr"]


x = np.arange(len(metrics))  # the label locations
width = 0.15  # the width of the bars


fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True, figsize=(7, 6))
axes = axes.flatten()


def make_subplot(data, ax, title):
    multiplier = -0.5
    for i, row in data.iterrows():
        offset = width * multiplier
        rects = ax.bar(
            x + offset,
            list(round(row[m], 2) for m in metrics),
            width,
            label=row["label"],
        )
        ax.bar_label(rects, padding=-16)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xticks(x + width, metrics)
    ax.yaxis.set_visible(False)
    ax.legend(loc="lower right", prop={"size": 12})


make_subplot(df[df.lang == "en"], axes[0], title="a. Event2012 (dataset in English)")
make_subplot(df[df.lang == "fr"], axes[1], title="b. Event2018 (dataset in French)")
plt.savefig("sbert_plot.pdf")
