import json
import numpy as np
from sklearn.metrics import ndcg_score
import os

do_eval = False

scores = [0, 0, 0]

ratios = [0.319, 0.005, 0.681]

if do_eval:
    with open("baseline/eval_scores.json") as f:
        scores[0] = json.loads(f.read())
    with open(f"result/lawformer-1-1e-5-0.1/eval_scores.json") as f:
        scores[1] = json.loads(f.read())
    with open(f"result/lawformer-8-1e-5-0.1/eval_scores.json") as f:
        scores[2] = json.loads(f.read())
else:
    with open("baseline/scores.json") as f:
        scores[0] = json.loads(f.read())
    with open(f"result/lawformer-1-1e-5-0.1/scores.json") as f:
        scores[1] = json.loads(f.read())
    with open(f"result/lawformer-8-1e-5-0.1/scores.json") as f:
        scores[2] = json.loads(f.read())

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
            np.array(scores[0][index]) * ratios[0]
            + np.array(scores[1][index]) * ratios[1]
            + np.array(scores[2][index]) * ratios[2]
        ).tolist()
        ndcg_scores.append(ndcg_score([y_true], [y_score], k=30))
    else:
        query = example["qid"]
        files = os.listdir(os.path.join("test/candidates", query))
        y_score = (
            np.array(scores[0][index]) * ratios[0]
            + np.array(scores[1][index]) * ratios[1]
            + np.array(scores[2][index]) * ratios[2]
        ).tolist()
        raw_rank_index = np.array(y_score).argsort().tolist()[::-1]
        result[query] = [int(files[i].split(".")[0]) for i in raw_rank_index]

if do_eval:
    print(f"eval_ndcg: {np.mean(ndcg_scores)}")
else:
    json.dump(
        result,
        open(
            os.path.join("comb_prediction.json"),
            "w",
            encoding="utf8",
        ),
        indent=2,
        ensure_ascii=False,
    )
