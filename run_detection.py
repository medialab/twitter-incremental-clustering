from twembeddings import build_matrix
from twembeddings import ClusteringAlgo

import csv
import yaml
import logging
import argparse
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score

from utils import METRICS_FILE

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--model',
                    nargs='+',
                    required=True,
                    choices=["sbert"],
                    help="""
                    One or several text embeddings
                    """
                    )
parser.add_argument('--dataset',
                    required=True,
                    help="""
                    Path to the dataset
                    """
                    )

parser.add_argument('--lang',
                    required=True,
                    choices=["en", "fr"])

parser.add_argument('--threshold',
                    type=float,
                    )

parser.add_argument('--window',
                    required=False,
                    type=int
                    )

parser.add_argument('--sub-model',
                    required=False,
                    type=str,
                    default="",
                    help="""
                    The name of HuggingFace sentence-BERT model
                    """
                    )

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)

def run(args: dict):
    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)
    for model in args["model"]:
        # load standard parameters
        params = dict(options["standard"])
        if model in options:
            # change standard parameters for this specific model
            for opt in options[model]:
                params[opt] = options[model][opt]
        for arg in args:
            if args[arg] is not None:
                # params from command line overwrite options.yaml file
                params[arg] = args[arg]

        params["model"] = model

        # Create document embeddings
        X, data = build_matrix(**params)

        # todo: improve window computation
        params["window"] = int(data.groupby("date").size().mean()*params["window"]/24// params["batch_size"] * params["batch_size"])

        clustering = ClusteringAlgo(threshold=float(params["threshold"]), window_size=params["window"], batch_size=params["batch_size"])
        clustering.add_vectors(X)

        # Run clustering algorithm
        y_pred = clustering.incremental_clustering()

        # Run evaluation
        ami = adjusted_mutual_info_score(data.label, y_pred)
        ari = adjusted_rand_score(data.label, y_pred)

        # Write detected labels to a csv file
        filename = params["dataset"].replace(".", "_clustering_results.")
        logging.info("Write predicted labels to {}".format(filename))
        data["pred"] = y_pred
        data[["id", "label", "pred"]].to_csv(filename, index=False, sep="\t", quoting=csv.QUOTE_ALL)

        # Write evaluation metrics to a csv file
        params.update({"AMI": ami, "ARI": ari})
        stats = pd.DataFrame(params, index=[0])
        stats = stats[["dataset", "model", "sub_model", "lang", "AMI", "ARI", "threshold", "window", "batch_size", "remove_mentions", "hashtag_split"]]
        print(stats[["sub_model", "threshold", "AMI", "ARI"]].iloc[0])

        try:
            results = pd.read_csv(METRICS_FILE)
        except FileNotFoundError:
            results = pd.DataFrame()
        stats = pd.concat([results, stats], ignore_index=True)
        stats.to_csv(METRICS_FILE, index=False)
        logging.info("Saved results to {}".format(METRICS_FILE))




if __name__ == '__main__':
    args = vars(parser.parse_args())
    run(args)