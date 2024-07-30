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

styler = table.style
styler.format(precision=2)
styler.format_index("\\textbf{{{}}}", escape="latex", axis=0)
styler.format_index("\\textbf{{{}}}", escape="latex", axis=1)


table_str = styler.to_latex(multirow_align="c", hrules=True, convert_css=True)

filename = METRICS_FILE.replace(".csv", ".tex")
print("Write latex table to {}".format(filename))
with open(filename, "w") as f:
    print(table_str.replace("metric", "").replace("model", "metric"), file=f)