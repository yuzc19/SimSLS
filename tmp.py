import json

with open("train/train.json", "r") as f:
    train_dataset = [json.loads(i) for i in f.readlines()]
sum = 0
for i in train_dataset:
    for j in i["processed_candidates"]:
        if j[1] == "3" or j[1] == "2":
            sum += 1
print(sum)
# with open("train/eval.json", "r") as f:
#     eval_dataset = [json.loads(i) for i in f.readlines()]

# for i in train_dataset:
#     print(i["qid"])
#     for j in eval_dataset:
#         if i["qid"] == j["qid"]:
#             exit(0)
# print("-----------------")
# for i in eval_dataset:
#     print(i["qid"])
# print("-----------------")
# print(len(train_dataset) + len(eval_dataset))

# from datasets import load_dataset, Dataset

# my_dict = {
#     "qid": [i["qid"] for i in train_dataset],
#     "query_text": [i["query_text"] for i in train_dataset],
#     "processed_candidates": [i["processed_candidates"] for i in train_dataset],
# }

# train_file = "train/eval.json"
# data_files = {}
# data_files["train"] = train_file
# extension = train_file.split(".")[-1]

# datasets = {}
# datasets["train"] = Dataset.from_dict(my_dict)
# print(datasets["train"].column_names)
# # # datasets = load_dataset(
# # #     extension, data_files=data_files, cache_dir="./data/", field="data"
# # # )
# print(datasets)
