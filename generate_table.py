import pandas as pd

from utils import METRICS_FILE

df = pd.read_csv(METRICS_FILE)

df["model"] = df.model.str.replace("_dataset", "")
df["dataset"] = df.dataset.str.replace(".tsv", "")

idx = df.groupby(['dataset', 'model'])['ARI'].transform('max') == df["ARI"]

ari = df[idx].pivot(index="dataset", columns="model", values="ARI")
ari["metric"] = "ARI"
ami = df[idx].pivot(index="dataset", columns="model", values="AMI")
ami["metric"] = "AMI"

table = pd.concat([ari, ami]).reset_index().sort_values("dataset").set_index(["dataset", "metric"])

styler = table.style.to_latex(multirow_align="c")

filename = METRICS_FILE.replace(".csv", ".tex")
print("Write latex table to {}".format(filename))
with open(filename, "w") as f:
    print(styler.replace("metric", "").replace("model", "metric"), file=f)