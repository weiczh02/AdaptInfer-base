#This code is for offline attention shift detection.

import os
import numpy as np
import ruptures as rpt
import json
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

dump_dir = "playground/13b_attn_dumps_t2v_mme"
save_dir = "playground/vis_13b"

results_f = {}
results_l = {}

algo_name = "binseg"    
cost_model = "l2"
sum = 1

first_cps_per_token = []

num = 0
file_list = [fn for fn in os.listdir(dump_dir) if fn.endswith(".npy")]
file_list = file_list[:1000]
top_k_tokens = 58

for fn in tqdm(file_list, desc="Processing files"):
    if not fn.endswith(".npy"):
        continue

    
    num +=1
    qid = fn.split("_")[0]
    arr = np.load(os.path.join(dump_dir, fn))  # shape = (L, V)
    L,V =arr.shape
    #print(L,V)
    token_attn_sums = arr.sum(axis=0)  # shape = (576,)
    top_token_indices = np.argsort(token_attn_sums)[-top_k_tokens:]
    
    if sum:
        arr = np.cumsum(arr, axis=0)

    for v in top_token_indices:
        curve = arr[:, v]  # shape (L,)
        penalty = np.log(L) * np.var(curve)
        #print(penalty)
        
        algo = rpt.Binseg(model=cost_model,jump=1).fit(curve)
        bkps = algo.predict(n_bkps=1)

        real_bkps = [cp-1 for cp in bkps]
        first_cp = real_bkps[0] if real_bkps else None
        first_cps_per_token.append(int(first_cp))


with open(os.path.join(save_dir,f"first_cps_histogram_sum_{algo_name}_{cost_model}.json"), "w") as f:
    json.dump(first_cps_per_token, f)

# reload
'''
with open("vtoken_first_cps.json") as f:
    results = json.load(f)
'''

print(f"{num} samples in total.")
counter = Counter(first_cps_per_token)

for layer in range(L):
    print(f"Layer {layer}: {counter.get(layer, 0)} tokens")


