from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import ndcg_score

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = AutoModel.from_pretrained("thunlp/Lawformer").cuda()


def get_embedding(text: str):
    inputs = tokenizer(
        text, return_tensors="pt"
    )  # It seems that the tokenizer automatically adds [CLS] (101) at the beginning.
    input_ids, token_type_ids, attention_mask = (
        inputs.input_ids.cuda(),
        inputs.token_type_ids.cuda(),
        inputs.attention_mask.cuda(),
    )
    outputs = model(input_ids, token_type_ids, attention_mask)
    return outputs.last_hidden_state[0][0].cpu().detach().numpy()  # 768
    # return outputs.pooler_output  # After training.


with open("train/candidates/4891/processed.json", "r") as f:
    data = json.load(f)
query_embedding = get_embedding(data["query_text"])
y_true, y_score = [], []
for _, label, candidate_text in tqdm(data["processed_candidates"]):
    y_true.append(label)
    if len(candidate_text) > 4000:
        candidate_text = candidate_text[:4000]
    candidate_embedding = get_embedding(candidate_text)
    similarity = 1 - cosine(query_embedding, candidate_embedding)
    y_score.append(similarity)

print(
    ndcg_score([y_true], [y_score], k=30)
)  # Best y_true (sorted by y_true) and y_true (sorted by y_score)
