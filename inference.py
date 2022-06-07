import argparse
from transformers import AutoTokenizer, AutoModel
from simcse.models import LawformerForCL
import torch
import json
from scipy.spatial.distance import cosine
from sklearn.metrics import ndcg_score
from copy import deepcopy
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--do_eval", type=bool, default=False)

args = parser.parse_args()
do_eval = args.do_eval

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# model = AutoModel.from_pretrained(
#     "thunlp/Lawformer"
# ).cuda()  # zero-shot: 0.6135746125773133
model_name_or_path = "result/lawformer-16-1e-5-0.1"
model = LawformerForCL.from_pretrained(model_name_or_path).cuda()


def batcher(sentences):
    batch = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt",
        max_length=3072,
        truncation=True,
        padding=True,
    )  # The tokenizer automatically adds [CLS] (101) at the beginning.
    for k in batch:
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(
            **batch, output_hidden_states=True, return_dict=True, sent_emb=True
        )
        pooler_output = outputs.pooler_output
    return pooler_output.cpu()


model.eval()

if do_eval:
    with open("train/eval.json", "r") as f:
        eval_dataset = [json.loads(i) for i in f.readlines()]
else:
    with open("test/test.json", "r") as f:
        eval_dataset = [json.loads(i) for i in f.readlines()]
y_scores = []
if do_eval:
    ndcg_scores = []
else:
    prediction = {}
for example in eval_dataset:
    texts = [example["query_text"]] + [i[2] for i in example["processed_candidates"]]
    pooler_output = None
    total = len(texts)
    for i in range(0, total, 8):
        pooler_output = (
            batcher([texts[i + j] for j in range(min(8, total - i))])
            if pooler_output == None
            else torch.cat(
                (
                    pooler_output,
                    batcher([texts[i + j] for j in range(min(8, total - i))]),
                )
            )
        )
    query_embedding = pooler_output[0].numpy()
    if do_eval:
        y_true, y_score = [int(i[1]) for i in example["processed_candidates"]], []
    else:
        y_score = []
    candidate_len = len(example["processed_candidates"])
    for i in range(1, candidate_len + 1):
        candidate_embedding = pooler_output[i].numpy()
        similarity = 1 - cosine(query_embedding, candidate_embedding)
        if do_eval:
            y_score.append(similarity)
        else:
            y_score.append([example["processed_candidates"][i - 1][0], similarity])
    if do_eval:
        y_scores.append(y_score)
        ndcg_scores.append(ndcg_score([y_true], [y_score], k=30))
    else:
        y_scores.append([x[1] for x in y_score])
        y_score.sort(key=lambda x: x[1], reverse=True)
        prediction[example["qid"]] = [y_score[i][0] for i in range(30)]

if do_eval:
    json.dump(y_scores, open(os.path.join(model_name_or_path, "eval_scores.json"), "w"))

    print(f"eval_ndcg: {np.mean(ndcg_scores)}")
else:
    json.dump(y_scores, open(os.path.join(model_name_or_path, "scores.json"), "w"))

    with open(model_name_or_path + "/prediction.json", "w") as f:
        f.write(json.dumps(prediction))
