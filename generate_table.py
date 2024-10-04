import pandas as pd
import numpy as np

from utils import METRICS_FILE

df = pd.read_csv(METRICS_FILE)
df = df.drop_duplicates()

df["model"] = df.model.str.replace("_dataset", "")
df["model"] = df.model.str.replace("sbert", "FSD-SBERT")
df["model"] = df.model.str.replace("HISEvent", "HE")
df["model"] = df.model.str.replace("twembeddings", "TW")
df["dataset"] = df.dataset.str.replace(".tsv", "")

idx = df.groupby(["dataset", "model"])["ARI"].transform("max") == df["ARI"]

ari = df[idx].pivot(index="dataset", columns="model", values="ARI")
ari["metric"] = "ARI"
ami = df[idx].pivot(index="dataset", columns="model", values="AMI")
ami["metric"] = "AMI"

table = (
    pd.concat([ari, ami])
    .reset_index()
    .sort_values("dataset")
    .set_index(["dataset", "metric"])
)

styler = table.style
styler.format(precision=2)
styler.format_index("\\textbf{{{}}}", escape="latex", axis=0)
styler.format_index("\\textbf{{{}}}", escape="latex", axis=1)


def highlight_max(s, props=""):
    return np.where(s == np.nanmax(s.values), props, "")


styler.apply(highlight_max, props="font-weight: bold", axis=1)

table_str = styler.to_latex(
    multirow_align="c",
    hrules=True,
    column_format="ll" + "c" * (table.shape[1]),
    convert_css=True,
)

filename = METRICS_FILE.replace(".csv", ".tex")
print("Write latex table to {}".format(filename))
with open(filename, "w") as f:
    print(table_str.replace("metric", "").replace("model", ""), file=f)
