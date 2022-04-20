import json
import os
from tqdm import tqdm

with open("train/query.json", "r") as f:
    query = [json.loads(l) for l in f]

with open("train/label_top30_dict.json", "r") as f:
    label_top30_dict = json.load(f)

i = 0
for qid, candidates in tqdm(label_top30_dict.items()):
    for j, q in enumerate(query):
        if str(q["ridx"]) == qid:
            i = j
            break
    assert str(query[i]["ridx"]) == qid
    query_text = query[i]["q"] + "涉及的罪名：" + "、".join(query[i]["crime"])
    processed_data = {}
    processed_data["qid"] = qid
    processed_data["query_text"] = query_text
    processed_data["processed_candidates"] = []
    candidates_dir = f"train/candidates/{qid}/"
    for filename in os.listdir(candidates_dir):
        if filename == "processed.json":
            continue
        cid = int(filename[:-5])
        with open(candidates_dir + filename, "r") as f:
            candidate = json.load(f)
            if "ajName" not in candidate:
                if "cpfxgc" not in candidate:
                    candidate_text = candidate["ajjbqk"]
                else:
                    candidate_text = candidate["ajjbqk"] + candidate["cpfxgc"]
            else:
                if "cpfxgc" not in candidate:
                    candidate_text = candidate["ajName"] + "：" + candidate["ajjbqk"]
                else:
                    candidate_text = (
                        candidate["ajName"]
                        + "："
                        + candidate["ajjbqk"]
                        + candidate["cpfxgc"]
                    )
            if str(cid) not in candidates or candidates[str(cid)] == 0:
                label = 0
            else:
                label = candidates[str(cid)]
            processed_data["processed_candidates"].append(
                [str(cid), label, candidate_text]
            )
    with open(candidates_dir + "processed.json", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(processed_data, ensure_ascii=False))
