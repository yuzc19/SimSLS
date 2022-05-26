import argparse
import json
import os
import random
from gensim.summarization import bm25
import jieba
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn import preprocessing

do_eval = False

parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument(
    "--input", type=str, default="train", help="input path of the dataset directory."
)
parser.add_argument(
    "--output", type=str, default="baseline", help="output path of the prediction file."
)

args = parser.parse_args()
input_path = args.input
if do_eval:
    input_query_path = os.path.join(input_path, "eval.json")
else:
    input_path = "test"
    input_query_path = os.path.join(input_path, "test.json")
input_candidate_path = os.path.join(input_path, "candidates")
output_path = args.output


def normalize(input):
    return (
        (
            preprocessing.MaxAbsScaler().fit_transform(np.array(input).reshape(-1, 1))
            * 2
            - 1
        )
        .reshape(1, -1)
        .tolist()[0]
    )  # [-1, 1]


if __name__ == "__main__":
    print("begin...")
    result = {}
    with open(os.path.join(os.path.dirname(__file__), "stopword.txt"), "r") as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend([".", "（", "）", "-"])

    lines = open(input_query_path, "r").readlines()
    y_scores = []
    if do_eval:
        ndcg_scores = []
    for line in lines:
        corpus = []
        query = str(eval(line)["qid"])
        # model init
        result[query] = []
        files = os.listdir(os.path.join(input_candidate_path, query))
        for index, file_ in enumerate(files):
            assert file_ == eval(line)["processed_candidates"][index][0] + ".json"
            file_json = json.load(
                open(os.path.join(input_candidate_path, query, file_), "r")
            )
            a = jieba.cut(file_json["ajjbqk"], cut_all=False)
            tem = " ".join(a).split()
            corpus.append([i for i in tem if not i in stopwords])
        bm25Model = bm25.BM25(corpus)

        # rank
        a = jieba.cut(eval(line)["query_text"], cut_all=False)
        tem = " ".join(a).split()
        q = [i for i in tem if not i in stopwords]
        if do_eval:
            y_true, y_score = [
                int(i[1]) for i in eval(line)["processed_candidates"]
            ], normalize(bm25Model.get_scores(q))
            ndcg_scores.append(ndcg_score([y_true], [y_score], k=30))
        else:
            y_score = normalize(bm25Model.get_scores(q))
        y_scores.append(y_score)
        raw_rank_index = np.array(y_score).argsort().tolist()[::-1]
        result[query] = [int(files[i].split(".")[0]) for i in raw_rank_index]
    if do_eval:
        json.dump(y_scores, open(os.path.join(output_path, "eval_scores.json"), "w"))
    else:
        json.dump(y_scores, open(os.path.join(output_path, "scores.json"), "w"))

    if do_eval:
        json.dump(
            result,
            open(
                os.path.join(output_path, "eval_prediction.json"), "w", encoding="utf8"
            ),
            indent=2,
            ensure_ascii=False,
        )
    else:
        json.dump(
            result,
            open(os.path.join(output_path, "prediction.json"), "w", encoding="utf8"),
            indent=2,
            ensure_ascii=False,
        )
    print("ouput done.")

    if do_eval:
        print(f"eval_ndcg: {np.mean(ndcg_scores)}")
