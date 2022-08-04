import pandas as pd
import argparse
import numpy as np
from ast import literal_eval
import json
from collections import defaultdict

MAX_N_COMMENTS = 10

if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Convert a CSV to a JSON of the post and comment ids to be used by the dataset explorer"
    )
    args.add_argument(
        "--csv",
        required=True,
        type=str,
        help="path to public VTC dataset csv (default: None)",
    )
    args.add_argument(
        "--output", default="dataset_ids.json", type=str, help="Output json file"
    )

    args = args.parse_args()

    df = pd.read_csv(args.csv)

    output = defaultdict(lambda: [])

    for loc, row in df.iterrows():
        reddit_id = np.base_repr(row.reddit_id, base=36).lower()
        comment_ids = [x[0] for x in literal_eval(row.comment_ids)[:MAX_N_COMMENTS]]
        output[row.subreddit].append([reddit_id, comment_ids])

    with open(args.output, "w") as f:
        json.dump(output, f)
