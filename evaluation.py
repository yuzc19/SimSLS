# zero-shot: 0.6135746125773133

from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import ndcg_score

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModel.from_pretrained("thunlp/Lawformer").cuda()
model.eval()


def get_embedding(text: str):
    inputs = tokenizer(
        text, return_tensors="pt"
    )  # The tokenizer automatically adds [CLS] (101) at the beginning.
    for k in inputs:
        inputs[k] = inputs[k].cuda()
    outputs = model(**inputs)
    return outputs.pooler_output[0].cpu().detach().numpy()  # 768
    # return outputs.pooler_output  # After training.


with open("train/eval.json", "r") as f:
    eval_dataset = [json.loads(i) for i in f.readlines()]
data = eval_dataset[0]
query_embedding = get_embedding(data["query_text"])
y_true, y_score = [], []
for _, label, candidate_text in tqdm(data["processed_candidates"]):
    y_true.append(int(label))
    if len(candidate_text) > 3072:
        candidate_text = candidate_text[:3072]
    candidate_embedding = get_embedding(candidate_text)
    similarity = 1 - cosine(query_embedding, candidate_embedding)
    y_score.append(similarity)

print(
    ndcg_score([y_true], [y_score], k=30)
)  # Best y_true (sorted by y_true) and y_true (sorted by y_score)
