import json
import numpy as np
import matplotlib.pyplot as plt
import data_prep as dp
from checkpoint import interactions


with open("final_test_results.json") as f:
    results = json.load(f)

names = list(results.keys())
means = [results[n]["mean"] for n in names]
stds = [results[n]["std"] for n in names]
times = [results[n]["time_sec"] for n in names]

order = np.argsort(means)
names = [names[i] for i in order]
means = [means[i] for i in order]
stds = [stds[i] for i in order]
times = [times[i] for i in order]


plt.figure(figsize=(10, 5))
bars = plt.bar(names, means)
plt.ylabel("nDCG@10 (mean ± std, 3 folds)")
plt.title("Final test performance")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, max(means) * 1.15)
for bar, m, s in zip(bars, means, stds):
    plt.text(bar.get_x() + bar.get_width()/2, m + s + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plot_test_ndcg.png", dpi=150)
plt.close()

#time vs nDCG scatter
plt.figure(figsize=(10, 8))
plt.scatter(times, means)
for n, t, m in zip(names, times, means):
    plt.annotate(n, (t, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
plt.xlabel("Training + scoring time (s, 3 folds)")
plt.ylabel("nDCG@10 (mean)")
plt.title("Cost vs. performance")
plt.tight_layout()
plt.savefig("plot_time_vs_ndcg.png", dpi=150)
plt.close()

# ---------- data characteristics ----------


per_user = interactions.groupby("user_id").size()
plt.figure(figsize=(8, 5))
plt.hist(per_user, bins=50)
plt.axvline(per_user.median(), color="red", linestyle="--", label=f"median = {int(per_user.median())}")
plt.xlabel("Interactions per user")
plt.ylabel("Number of users")
plt.title("Per-user interaction distribution")
plt.legend()
plt.tight_layout()
plt.savefig("plot_user_distribution.png", dpi=150)
plt.close()



print("saved")