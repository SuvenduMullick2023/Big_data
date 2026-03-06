import numpy as np
import matplotlib.pyplot as plt

def lsh_probability(s, r, b):
    return 1 - (1 - s**r)**b

# similarity values
s = np.linspace(0, 1, 100)

configs = [
    (5,10,"50 hashes (r=5,b=10)"),
    (5,20,"100 hashes (r=5,b=20)"),
    (5,40,"200 hashes (r=5,b=40)"),
    (10,20,"200 hashes (r=10,b=20)")
]

plt.figure(figsize=(8,6))

for r,b,label in configs:
    prob = lsh_probability(s,r,b)
    plt.plot(s,prob,label=label)

# similarity threshold
plt.axvline(0.6, linestyle="--", label="Threshold = 0.6")

plt.xlabel("Jaccard Similarity (s)")
plt.ylabel("Probability of becoming candidate")
plt.title("LSH S-Curve")
plt.legend()
plt.grid(True)
plt.savefig("/home/suvendu/mlbd/code/Big_data/lsh_s_curve.png")
plt.show()