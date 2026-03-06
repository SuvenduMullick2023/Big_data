import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract,collect_list, concat_ws ,col,count,desc,length, avg,stddev,to_date
import random
import itertools
import matplotlib.pyplot as plt
import pandas as pd        



################################################
#Min-Hashing on MovieLens dataset
################################################

from collections import defaultdict
import csv

from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

spark = SparkSession.builder \
    .appName("MovieLens100K") \
    .getOrCreate()

# Path to ml-100k folder
path = "/home/suvendu/mlbd/ml-100k/"

# Load u.data (tab separated, no header)
ratings = spark.read.csv(
    path + "u.data",
    sep="\t",
    header=False,
    inferSchema=True
)

# Rename columns
ratings = ratings.toDF("userId", "movieId", "rating", "timestamp")

# Keep only userId and movieId
user_movies_rdd = ratings.select("userId", "movieId") \
    .rdd \
    .map(lambda x: (x["userId"], x["movieId"])) \
    .groupByKey() \
    .mapValues(set)

user_movies = dict(user_movies_rdd.collect())
# Collect to Python dictionary (safe for 943 users)

print(type(user_movies))

users = list(user_movies.keys())
print("Number of users:", len(user_movies))
# List of users

print("Movies rated by first user:", len(user_movies[users[0]]))

for user, movies in itertools.islice(user_movies.items(), 5):
    print("User:", user)
    print("Number of movies:", len(movies))
    print("Some movies:", list(movies)[:10])
    print("-" * 40)
    
#sample = list(user_movies.items())[:5]
#print("Print first 5 users movies " , sample)    

for user in users[:5]:
    print(f"User {user} -> {len(user_movies[user])} movies")

movie_counts = [len(movies) for movies in user_movies.values()]

print("Min movies rated:", min(movie_counts))
print("Max movies rated:", max(movie_counts))
print("Average movies rated:", sum(movie_counts)/len(movie_counts))
    

# ---Exact Jaccard
def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

exact_sim = {}
for u1, u2 in itertools.combinations(users, 2):
    sim = jaccard(user_movies[u1], user_movies[u2])
    exact_sim[(u1,u2)] = sim

ground_truth_05 = {pair for pair,sim in exact_sim.items() if sim >= 0.5}
ground_truth_06 = {pair for pair,sim in exact_sim.items() if sim >= 0.6}
ground_truth_08 = {pair for pair,sim in exact_sim.items() if sim >= 0.8}

print("ground_truth with sim >= 0.5 ",ground_truth_05)
print("ground_truth with sim >= 0.6 ",ground_truth_06)
print("ground_truth with sim >= 0.8 ",ground_truth_08)

# Hash Function Generator
def generate_hash_funcs(num_hashes, max_val):
    hash_funcs = []
    for _ in range(num_hashes):
        a = random.randint(1, max_val)
        b = random.randint(0, max_val)
        hash_funcs.append((a, b))
    return hash_funcs

# --MinHash
def compute_signature(movie_set, hash_funcs, m):
    signature = []
    for a,b in hash_funcs:
        min_hash = min(((a*x + b) % 1000003) % m for x in movie_set)
        signature.append(min_hash)
    return signature

# Estimated Similarity
def est_sim(sig1, sig2):
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)


# -- Run MinHash Experiment
def run_minhash_experiment(num_hashes, threshold):
    max_movie_id = max(
        movie for movies in user_movies.values() for movie in movies
    )

    hash_funcs = generate_hash_funcs(num_hashes, max_movie_id)

    signatures = {}
    for u in users:
        signatures[u] = compute_signature(user_movies[u], hash_funcs, max_movie_id)

    estimated_pairs = set()
    for u1, u2 in itertools.combinations(users, 2):
        sim = est_sim(signatures[u1], signatures[u2])
        if sim >= threshold:
            estimated_pairs.add((u1, u2))

    return estimated_pairs

# -- Compute FP/FN  Errors
def compute_errors(estimated, ground_truth):
    fp = len(estimated - ground_truth)
    fn = len(ground_truth - estimated)
    return fp, fn

#for gnd_truth in [ground_truth_05,ground_truth_06,ground_truth_08]:
results = []
# --Run 5 Trials
for t in [50,100,200]:
    fp_list = []
    fn_list = []

    for run in range(5):
        # set different seed per run for reproducibility
        random.seed(run)
        est = run_minhash_experiment(t, 0.5)
        fp, fn = compute_errors(est, ground_truth_05)
        fp_list.append(fp)
        fn_list.append(fn)

        print(f"Run {run+1}: FP={fp}, FN={fn}")

    print(f"\nHashes = {t}")
    print("Average FP:", sum(fp_list)/5)
    print("Average FN:", sum(fn_list)/5)
    print("-" * 50)

# Convert to DataFrame
df = pd.DataFrame(results)

print("\nFull Results DataFrame:")
print(df)

# Analysis of MinHash Results

import matplotlib.pyplot as plt
import pandas as pd
import statistics

# Raw results
data = {
    50: {"FP": [148,119,611,115,398], "FN": [3,2,2,4,1]},
    100: {"FP": [74,30,80,129,70], "FN": [3,2,1,1,3]},
    200: {"FP": [44,17,54,45,32], "FN": [3,2,2,0,0]}
}

# Compute summary statistics
summary = []
for hashes, values in data.items():
    avg_fp = statistics.mean(values["FP"])
    avg_fn = statistics.mean(values["FN"])
    std_fp = statistics.stdev(values["FP"])
    std_fn = statistics.stdev(values["FN"])
    
    summary.append([hashes, avg_fp, avg_fn, std_fp, std_fn])

df = pd.DataFrame(summary, columns=["Hashes", "Avg FP", "Avg FN", "Std FP", "Std FN"])
# Plot 1: Average False Positives vs Hashes
plt.figure()
plt.plot(df["Hashes"], df["Avg FP"])
plt.xlabel("Number of Hash Functions")
plt.ylabel("Average False Positives")
plt.title("Average False Positives vs Number of Hash Functions")
plt.savefig("/home/suvendu/mlbd/code/Big_data/avg_fp_vs_fn.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Average False Negatives vs Hashes
plt.figure()
plt.plot(df["Hashes"], df["Avg FN"])
plt.xlabel("Number of Hash Functions")
plt.ylabel("Average False Negatives")
plt.title("Average False Negatives vs Number of Hash Functions")
plt.savefig("/home/suvendu/mlbd/code/Big_data/avg_fn.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Standard Deviation of False Positives vs Hashes
plt.figure()
plt.plot(df["Hashes"], df["Std FP"])
plt.xlabel("Number of Hash Functions")
plt.ylabel("Std Deviation of False Positives")
plt.title("Variance Reduction with More Hash Functions")

plt.savefig("/home/suvendu/mlbd/code/Big_data/avg_fp.png", dpi=300, bbox_inches='tight')
plt.show()


for gnd_truth in [ground_truth_05,ground_truth_06,ground_truth_08]:
    results = []
    # --Run 5 Trials
    for t in [50,100,200]:
        fp_list = []
        fn_list = []

        for run in range(5):
            # set different seed per run for reproducibility
            random.seed(run)
            est = run_minhash_experiment(t, 0.5)
            fp, fn = compute_errors(est, gnd_truth)
            fp_list.append(fp)
            fn_list.append(fn)

            print(f"Run {run+1}: FP={fp}, FN={fn}")

        print(f"\nHashes = {t}")
        print("Average FP:", sum(fp_list)/5)
        print("Average FN:", sum(fn_list)/5)
        print("-" * 50)
      
spark.stop()    
