import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract,collect_list, concat_ws ,col,count,desc,length, avg,stddev,to_date
import random
import itertools




spark = SparkSession.builder \
    .appName("MinHash_LSH_Assignment") \
    .config("spark.local.dir", "/home/suvendu/spark-temp") \
    .getOrCreate()



sc = spark.sparkContext

#df_spark = spark.read.option('header','true').text('/home/suvendu/mlbd/D184MB/200.txt')
# -------------------------------------------------
# Load Documents
# -------------------------------------------------
docs = {
    "D1": sc.textFile("/home/suvendu/mlbd/minhash/D1.txt").collect()[0],
    "D2": sc.textFile("/home/suvendu/mlbd/minhash/D2.txt").collect()[0],
    "D3": sc.textFile("/home/suvendu/mlbd/minhash/D3.txt").collect()[0],
    "D4": sc.textFile("/home/suvendu/mlbd/minhash/D4.txt").collect()[0],
}

# -------------------------------------------------
# K-GRAM GENERATORS
# -------------------------------------------------

def char_kgrams(text, k):
    return set([text[i:i+k] for i in range(len(text) - k + 1)])

def word_kgrams(text, k):
    words = text.split()
    return set([" ".join(words[i:i+k]) for i in range(len(words) - k + 1)])

# -------------------------------------------------
# Build K-gram Sets
# -------------------------------------------------

char2 = {doc: char_kgrams(text, 2) for doc, text in docs.items()}
char3 = {doc: char_kgrams(text, 3) for doc, text in docs.items()}
word2 = {doc: word_kgrams(text, 2) for doc, text in docs.items()}

# -------------------------------------------------
# Jaccard Similarity
# -------------------------------------------------

def jaccard(A, B):
    return len(A & B) / len(A | B)

def compute_all_pairs(gram_dict):
    pairs = list(itertools.combinations(gram_dict.keys(), 2))
    results = {}
    for a, b in pairs:
        results[(a, b)] = jaccard(gram_dict[a], gram_dict[b])
    return results


# -------------------------------------------------
# Part B: Exact Jaccard Similarities
# -------------------------------------------------

print("Character 2-grams Jaccard:")
char2_sim = compute_all_pairs(char2)
print(char2_sim)

print("\nCharacter 3-grams Jaccard:")
char3_sim = compute_all_pairs(char3)
print(char3_sim)

print("\nWord 2-grams Jaccard:")
word2_sim = compute_all_pairs(word2)
print(word2_sim)

# -------------------------------------------------
# MIN-HASH
# -------------------------------------------------

def generate_hash_functions(num_hash, max_shingle):
    hash_funcs = []
    for _ in range(num_hash):
        a = random.randint(1, max_shingle-1)
        b = random.randint(0, max_shingle-1)
        hash_funcs.append((a, b))
    return hash_funcs

def minhash_signature(shingle_set, hash_funcs, max_shingle):
    signature = []
    for a, b in hash_funcs:
        min_hash = float('inf')
        for shingle in shingle_set:
            x = abs(hash(shingle)) % max_shingle
            h = (a * x + b) % max_shingle
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature

def estimate_similarity(sig1, sig2):
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)

# -------------------------------------------------
# Part A: MinHash Approximation for D1 and D2
# Using 3-grams
# -------------------------------------------------

t_values = [20, 60, 150, 300, 600]
max_shingle = 10007  # large prime


print("\nMinHash Approximation (3-grams) for D1 vs D2:")
for t in t_values:
    hash_funcs = generate_hash_functions(t, max_shingle)

    sig1 = minhash_signature(char3["D1"], hash_funcs, max_shingle)
    sig2 = minhash_signature(char3["D2"], hash_funcs, max_shingle)

    est_sim = estimate_similarity(sig1, sig2)

    print(f"t = {t}, Estimated Jaccard = {round(est_sim,4)}")
    

print("\nMinHash Approximation (3-grams) for ALL document pairs:")

for t in t_values:
    print(f"\n--- t = {t} ---")

    # Generate hash functions
    hash_funcs = generate_hash_functions(t, max_shingle)

    # Step 1: Compute signatures for all documents
    signatures = {}
    for doc in char3.keys():
        signatures[doc] = minhash_signature(char3[doc], hash_funcs, max_shingle)

    # Step 2: Compute similarity for all pairs
    for doc1, doc2 in itertools.combinations(char3.keys(), 2):
        est_sim = estimate_similarity(signatures[doc1], signatures[doc2])
        print(f"{doc1} vs {doc2} -> {round(est_sim,4)}") 
        
        
        
import itertools
import math

# -----------------------------------------
# 3, LSH Parameters
# -----------------------------------------
t = 160
r = 8
b = 20

print("\nLSH PARAMETERS")
print(f"t = {t}, r = {r}, b = {b}")
print("Threshold approx =", (1/b)**(1/r))

# -----------------------------------------
# LSH Probability Function
# f(s) = 1 - (1 - s^r)^b
# -----------------------------------------
def lsh_probability(s, r, b):
    return 1 - (1 - (s ** r)) ** b

# -----------------------------------------
# Use stable similarities (e.g., from t=600)

# -----------------------------------------

t=600 

Dict_similarity ={}
# Step 1: Compute signatures for all documents
signatures = {}
for doc in char3.keys():
    signatures[doc] = minhash_signature(char3[doc], hash_funcs, max_shingle)

# Step 2: Compute similarity for all pairs
for doc1, doc2 in itertools.combinations(char3.keys(), 2):
    est_sim = estimate_similarity(signatures[doc1], signatures[doc2])
    print(f"{doc1} vs {doc2} -> {round(est_sim,4)}")
    Dict_similarity[doc1 , doc2]= round(est_sim,4)
 
print(Dict_similarity)   
'''similarities = {
    ("D1","D2"): 0.975,
    ("D1","D3"): 0.6017,
    ("D1","D4"): 0.3117,
    ("D2","D3"): 0.5917,
    ("D2","D4"): 0.31,
    ("D3","D4"): 0.3333
}'''

print("\nProbability of Being Candidate Pair (LSH):")

for pair, s in Dict_similarity.items():
    prob = lsh_probability(s, r, b)
    print(f"{pair[0]} vs {pair[1]} -> {prob:.4f}")           



################################################
#Min-Hashing on MovieLens dataset
################################################

from collections import defaultdict
import csv

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MovieLens").getOrCreate()
path ='/home/suvendu/mlbd/ml-32m/'
ratings = spark.read.csv(path + "ratings.csv", header=True, inferSchema=True)

user_movies = ratings.select("userId", "movieId") \
                      .rdd \
                      .map(lambda x: (x["userId"], x["movieId"])) \
                      .groupByKey() \
                      .mapValues(set)

#user_movies = defaultdict(set)

'''with open("u.data", "r") as f:
    for line in f:
        user, movie, rating, timestamp = line.strip().split("\t")
        user_movies[int(user)].add(int(movie))'''

#users = list(user_movies.keys())


print(len(user_movies))
print(len(user_movies[1]))

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

# --MinHash
def compute_signature(movie_set, hash_funcs, m):
    signature = []
    for a,b in hash_funcs:
        min_hash = min(((a*x + b) % 1000003) % m for x in movie_set)
        signature.append(min_hash)
    return signature

def est_sim(sig1, sig2):
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)


# -- Run MinHash Experiment
def run_minhash_experiment(num_hashes, threshold):
    max_movie_id = 2000
    hash_funcs = generate_hash_funcs(num_hashes, max_movie_id)

    signatures = {}
    for u in users:
        signatures[u] = compute_signature(user_movies[u], hash_funcs, max_movie_id)

    estimated_pairs = set()
    for u1, u2 in itertools.combinations(users, 2):
        sim = est_sim(signatures[u1], signatures[u2])
        if sim >= threshold:
            estimated_pairs.add((u1,u2))

    return estimated_pairs

# -- Compute FP/FN
def compute_errors(estimated, ground_truth):
    fp = len(estimated - ground_truth)
    fn = len(ground_truth - estimated)
    return fp, fn

# --Run 5 Trials
for t in [50,100,200]:
    total_fp = 0
    total_fn = 0

    for _ in range(5):
        est = run_minhash_experiment(t, 0.5)
        fp, fn = compute_errors(est, ground_truth_05)
        total_fp += fp
        total_fn += fn

    print(f"\nHashes = {t}")
    print("Avg FP:", total_fp/5)
    print("Avg FN:", total_fn/5)
    
spark.stop()    
# PART 2 — LSH Implementation
    