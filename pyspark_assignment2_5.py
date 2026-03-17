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



def generate_hash_funcs(num_hashes, max_movie_id):
    hash_funcs = []
    for i in range(num_hashes):
        a = random.randint(1, max_movie_id)
        b = random.randint(0, max_movie_id)
        hash_funcs.append((a, b))
    return hash_funcs


def compute_signature(movies, hash_funcs, max_movie_id):
    signature = []
    for a, b in hash_funcs:
        min_hash = min([(a * m + b) % max_movie_id for m in movies])
        signature.append(min_hash)
    return signature

def LSH(signatures, r, b):
    candidates = set()
    
    for band in range(b):
        buckets = {}
        
        for userId, sig in signatures.items():   # FIXED
            start = band * r
            end = start + r
            band_sig = tuple(sig[start:end])
            
            if band_sig not in buckets:
                buckets[band_sig] = []
            buckets[band_sig].append(userId)
        
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidates.add(tuple(sorted(pair)))
    
    return candidates

'''def LSH(signatures, r, b):
    candidates = set()
    
    for band in range(b):
        buckets = {}
        
        for userId, sig in signatures:
            start = band * r
            end = start + r
            band_sig = tuple(sig[start:end])
            
            if band_sig not in buckets:
                buckets[band_sig] = []
            buckets[band_sig].append(userId)
        
        for bucket in buckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidates.add(tuple(sorted(pair)))
    
    return candidates'''


def run_experiment(num_hashes, r, b, threshold):
    max_movie_id = 2000
    hash_funcs = generate_hash_funcs(num_hashes, max_movie_id)
    
    '''signatures = user_rdd.map(
        lambda x: (x[0], compute_signature(x[1], hash_funcs, max_movie_id))
    ).collect()'''
    
    signatures = {}
    for u in users:
        signatures[u] = compute_signature(user_movies[u], hash_funcs, max_movie_id)
    
    candidates = LSH(signatures, r, b)
    
    if threshold == 0.6:
        truth = ground_truth_06
    else:
        truth = ground_truth_08
    
    false_pos = len(candidates - truth)
    false_neg = len(truth - candidates)
    
    return false_pos, false_neg



def average_runs(num_hashes, r, b, threshold):
    fp_list = []
    fn_list = []
    
    for run in range(5):
        random.seed(run)
        fp, fn = run_experiment(num_hashes, r, b, threshold)
        fp_list.append(fp)
        fn_list.append(fn)
    
    return sum(fp_list)/5, sum(fn_list)/5

for run in range(5):
    random.seed(run)
    print(f"Run {run+1}")
    print("For threshold = 0.6","-"*50) 
    print("50 hashes (r=5, b=10):", average_runs(50, 5, 10, 0.6))

    print("100 hashes (r=5, b=20):", average_runs(100, 5, 20, 0.6))

    print("200 hashes (r=5, b=40):", average_runs(200, 5, 40, 0.6))

    print("200 hashes (r=10, b=20):", average_runs(200, 10, 20, 0.6))

# For threshold = 0.8
    print("For threshold = 0.8","-"*50) 
    print("50 hashes (r=5, b=10):", average_runs(50, 5, 10, 0.8))
    print("100 hashes (r=5, b=20):", average_runs(100, 5, 20, 0.8))
    print("200 hashes (r=5, b=40):", average_runs(200, 5, 40, 0.8))
    print("200 hashes (r=10, b=20):", average_runs(200, 10, 20, 0.8))


spark.stop()


