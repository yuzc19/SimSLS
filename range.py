import json
import numpy as np
from sklearn.metrics import ndcg_score
import os

do_eval = True

index = 0

model_name_or_paths = [
    "result/lawformer-1-1e-5-0.1",
    "result/lawformer-8-1e-5-0.1",
    "result/lawformer",
]
model_name_or_path = model_name_or_paths[index]

BM25_ratios = [0.061, 0.319, 0.39]  # 0.219 for new
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
select = []
for r in range(1001):
    ratio = r * 0.001
    if do_eval:
        ndcg_scores = []
    else:
        result = {}
    for index, example in enumerate(eval_dataset):
        if do_eval:
            y_true, y_score = [int(i[1]) for i in example["processed_candidates"]], (
                np.array(BM25_scores[index]) * ratio
                + np.array(model_scores[index]) * (1 - ratio)
            ).tolist()
            ndcg_scores.append(ndcg_score([y_true], [y_score], k=30))

    if do_eval:
        select.append([ratio, np.mean(ndcg_scores)])

select.sort(key=lambda x: x[1], reverse=True)

print(select[0])
