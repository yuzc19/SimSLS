import argparse
import json
import numpy as np
from sklearn.metrics import ndcg_score
import os

parser = argparse.ArgumentParser()

parser.add_argument("--do_eval", type=bool, default=False)

args = parser.parse_args()
do_eval = args.do_eval

index = 2

model_name_or_paths = [
    "result/lawformer-1-1e-5-0.1",
    "result/lawformer-8-1e-5-0.1",
    "result/lawformer-16-1e-5-0.1",
]
model_name_or_path = model_name_or_paths[index]

BM25_ratios = [0.083, 0.314, 0.393]
BM25_ratio = BM25_ratios[index]

if do_eval:
    with open("baseline/eval_scores.json") as f:
        BM25_scores = json.loads(f.read())
    with open(f"{model_name_or_path}/eval_scores.json") as f:
        model_scores = json.loads(f.read())
else:
    with open("baseline/scores.json") as f:
        BM25_scores = json.loads(f.read())
    with open(f"{model_name_or_path}/scores.json") as f:
        model_scores = json.loads(f.read())

if do_eval:
    with open("train/eval.json", "r") as f:
        eval_dataset = [json.loads(i) for i in f.readlines()]
else:
    with open("test/test.json", "r") as f:
        eval_dataset = [json.loads(i) for i in f.readlines()]
if do_eval:
    ndcg_scores = []
else:
    result = {}
for index, example in enumerate(eval_dataset):
    if do_eval:
        y_true, y_score = [int(i[1]) for i in example["processed_candidates"]], (
            np.array(BM25_scores[index]) * BM25_ratio
            + np.array(model_scores[index]) * (1 - BM25_ratio)
        ).tolist()
        ndcg_scores.append(ndcg_score([y_true], [y_score], k=30))
    else:
        query = example["qid"]
        files = os.listdir(os.path.join("test/candidates", query))
        y_score = (
            np.array(BM25_scores[index]) * BM25_ratio
            + np.array(model_scores[index]) * (1 - BM25_ratio)
        ).tolist()
        raw_rank_index = np.array(y_score).argsort().tolist()[::-1]
        result[query] = [int(files[i].split(".")[0]) for i in raw_rank_index]

if do_eval:
    print(f"eval_ndcg: {np.mean(ndcg_scores)}")
else:
    json.dump(
        result,
        open(
            os.path.join(model_name_or_path, "comb_prediction.json"),
            "w",
            encoding="utf8",
        ),
        indent=2,
        ensure_ascii=False,
    )
