from twembeddings import build_matrix
from twembeddings import ClusteringAlgo, ClusteringAlgoSparse

import yaml
import argparse
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, rand_score, mutual_info_score

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--model',
                    nargs='+',
                    required=True,
                    choices=["sbert", "tfidf_dataset"],
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
                    nargs='+',
                    required=False
                    )

parser.add_argument('--window',
                    required=False,
                    default=24,
                    type=int
                    )

parser.add_argument('--remove_mentions',
                    action='store_true'
                    )

parser.add_argument('--sub-model',
                    required=False,
                    type=str
                    )

def run(args):
    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)
    for model in args["model"]:
        # load standard parameters
        params = options["standard"]
        if model in options:
            # change standard parameters for this specific model
            for opt in options[model]:
                params[opt] = options[model][opt]
        for arg in args:
            if args[arg] is not None:
                # params from command line overwrite options.yaml file
                params[arg] = args[arg]

        params["model"] = model

        X, data = build_matrix(**params)
        # Improve window computation


if __name__ == '__main__':
    args = vars(parser.parse_args())
    run(args)