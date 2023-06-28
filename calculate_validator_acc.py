import json
import numpy as np
import os
import pandas as pd

# set precision to 2
np.set_printoptions(precision=2)


def parse_pred(pred):
    if ":" in pred:
        last_colon_idx = pred.rindex(":")
        pred = pred[last_colon_idx + 1 :]

    return "yes" in pred.strip().lower()


def calculate_acc(path):
    with open(path, "r") as f:
        data = json.load(f)

    predictions = [parse_pred(x["generations"][0]["lm_postprocess"]) for x in data]

    gold_labels = ["yes" in x["demonstration"].lower() for x in data]

    group2acc = {}
    for is_bert in [True, False]:
        group = "bert_factor" if is_bert else "others"
        idxes = [i for i, x in enumerate(data) if x["orig_d"]["is_bert"] == is_bert]

        group2acc[group] = np.mean([predictions[i] == gold_labels[i] for i in idxes])

    return group2acc


if __name__ == "__main__":
    # path = "mount/models/koala_13b_v2-0621_all_verifier_data_koala/temperature=0.00_n=1_step=8000.json"
    # path = "mount/models/llama_30B-0621_all_verifier_data_koala/temperature=0.00_n=1_step=0.json"
    # path = "mount/models/llama_13-base-0621_all_verifier_data_koala/temperature=0.00_n=1_step=5000.json"
    # path = "mount/models/lr_0.0002-llama_13-base-0621_all_verifier_data_koala/temperature=0.00_n=1_step=0.json"
    # path = "mount/models/llama_13-base-0621_all_verifier_data_koala/temperature=0.00_n=1_step=10000.json"

    for step in [0, 5000, 10000, 15000, 20000]:
        path = f"mount/models/google_flan-t5-xxl_0621_all_verifier_data_t5_0/temperature=0.00_n=1_step={step}.json"
        print(calculate_acc(path))
    exit(0)

    for step in [0, 5000]:
        path = f"mount/models/lr_0.0002-llama_13-base-0621_all_verifier_data_koala/temperature=0.00_n=1_step={step}.json"
        print(calculate_acc(path))
    exit(0)
    for step in np.arange(0, 15001, 5000):
        path = f"mount/models/llama_30B-0621_all_verifier_data_koala/temperature=0.00_n=1_step={step}.json"
        print(calculate_acc(path))
    exit(0)
    for size in ["xl", "xxl"]:
        print(size)
        step2key2acc = {}
        for step in np.arange(0, 10001, 500):
            path = f"mount/models/google_flan-t5-{size}_0615_all_verifier_data_0/temperature=0.00_n=1_step={step}.json"
            if not os.path.exists(path):
                continue
            step2key2acc[step] = calculate_acc(path)
        df = pd.DataFrame(step2key2acc).T
        print(df)
