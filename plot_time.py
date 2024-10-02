import pandas as pd
from matplotlib import pyplot as plt
from labellines import labelLine, labelLines

from utils import METRICS_FILE

df = pd.read_csv(METRICS_FILE)
df = df.drop_duplicates()
df = df[(df.threshold == 0.45) & (df.lang == "en")].sort_values("batch_size")


fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True, figsize=(6, 8))

axes = axes.flatten()

time_plot = axes[0]
time_plot_x = list(df.batch_size.values)
time_plot_y = list(df.seconds.values)
time_plot.set_axisbelow(True)
time_plot.grid()
time_plot.scatter(time_plot_x, time_plot_y, label="Processing time", color="k")
time_plot.plot(time_plot_x, time_plot_y, label="Processing time", color="k")
time_plot.set_xticklabels([])
time_plot.set_ylabel("execution time (seconds)")


metrics_plot = axes[1]
metrics_plot.set_ylim([0.5, 1])
metrics_plot.set_axisbelow(True)
metrics_plot.grid()
# for metric in ["AMI", "f1", "ARI"]:
for metric in ["AMI"]:
    metrics_plot.scatter(
        time_plot_x, list(df[metric].values), label=metric.upper(), color="k"
    )
    metrics_plot.plot(
        time_plot_x, list(df[metric].values), label=metric.upper(), color="k"
    )
# labelLines(metrics_plot.get_lines(), align=False, fontsize=14)
metrics_plot.set_ylabel("adjusted mutual information")
metrics_plot.set_xlabel("batch size (number of documents)")

plt.savefig("timeplot.png")
