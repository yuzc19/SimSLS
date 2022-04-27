# zero-shot: 0.6135746125773133

from transformers import AutoTokenizer, AutoModel
from simcse.models import LawformerForCL
import torch
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import ndcg_score
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
# model = AutoModel.from_pretrained("thunlp/Lawformer").cuda()
model = LawformerForCL.from_pretrained("result/lawformer-1-5e-5-0.1").cuda()


# def get_embedding(text: str):
#     inputs = tokenizer(
#         text, return_tensors="pt"
#     )  # The tokenizer automatically adds [CLS] (101) at the beginning.
#     for k in inputs:
#         inputs[k] = inputs[k].cuda()
#     outputs = model(**inputs)
#     return outputs.pooler_output[0].cpu().detach().numpy()  # 768
#     # return outputs.pooler_output  # After training.


# with open("train/eval.json", "r") as f:
#     eval_dataset = [json.loads(i) for i in f.readlines()]
# data = eval_dataset[0]
# query_embedding = get_embedding(data["query_text"])
# y_true, y_score = [], []
# for _, label, candidate_text in tqdm(data["processed_candidates"]):
#     y_true.append(int(label))
#     if len(candidate_text) > 3072:
#         candidate_text = candidate_text[:3072]
#     candidate_embedding = get_embedding(candidate_text)
#     similarity = 1 - cosine(query_embedding, candidate_embedding)
#     y_score.append(similarity)

# print(
#     ndcg_score([y_true], [y_score], k=30)
# )  # Best y_true (sorted by y_true) and y_true (sorted by y_score)


def batcher(sentences):
    batch = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt",
        max_length=3072,
        truncation=True,
        padding=True,
    )
    for k in batch:
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(
            **batch, output_hidden_states=True, return_dict=True, sent_emb=True
        )
        pooler_output = outputs.pooler_output
    return pooler_output.cpu()


model.eval()

with open("train/eval.json", "r") as f:
    eval_dataset = [json.loads(i) for i in f.readlines()]
ndcg_scores = []
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
    y_true, y_score = [int(i[1]) for i in example["processed_candidates"]], []
    candidate_len = len(example["processed_candidates"])
    for i in range(1, candidate_len + 1):
        candidate_embedding = pooler_output[i].numpy()
        similarity = 1 - cosine(query_embedding, candidate_embedding)
        y_score.append(similarity)
    ndcg_scores.append(ndcg_score([y_true], [y_score], k=30))

print(np.mean(ndcg_scores))
