# ============================================
# ITEM BASED COLLABORATIVE FILTERING
# FINAL SUBMISSION VERSION
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct
from pyspark.sql import Row
import numpy as np
import time
from math import sqrt

spark = SparkSession.builder \
    .appName("ItemCF_Final") \
    .config("spark.driver.memory","6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

path="/home/suvendu/mlbd/ml-latest-small/"

start=time.time()

# ---------------------------
# 1 Load Data + Train-Test Split
# ---------------------------

print("\nLoading data...")

ratings = spark.read.csv(path+"ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv(path+"movies.csv", header=True, inferSchema=True)

train, test = ratings.randomSplit([0.8,0.2], seed=42)

print("Train:", train.count())
print("Test:", test.count())

# ---------------------------
# 2 Build Item Rating Matrix
# ---------------------------

print("\nBuilding item rating matrix...")

item_dict = {
    row["movieId"]: {r["userId"]: r["rating"] for r in row["ratings"]}
    for row in train.groupBy("movieId")
    .agg(collect_list(struct("userId","rating")).alias("ratings"))
    .collect()
}

items = list(item_dict.keys())

# ---------------------------
# 3 Cosine Similarity
# ---------------------------

def cosine(i1, i2):

    common = set(i1.keys()) & set(i2.keys())

    if len(common) < 2:
        return 0

    v1 = np.array([i1[u] for u in common])
    v2 = np.array([i2[u] for u in common])

    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)

    return num/den if den != 0 else 0


# ---------------------------
# 4 Compute Item Similarity (Top-K)
# ---------------------------

print("\nComputing item similarity...")

K_SIM = 10
item_neighbors = {}

for i, m1 in enumerate(items):

    if i % 200 == 0:
        print("Processing item:", i)

    sims = []

    for m2 in items:

        if m1 == m2:
            continue

        sim = cosine(item_dict[m1], item_dict[m2])

        if sim > 0:
            sims.append((m2, sim))

    sims = sorted(sims, key=lambda x:x[1], reverse=True)[:K_SIM]

    item_neighbors[m1] = sims

print("Item similarity ready")

# ---------------------------
# 5 Build User Dictionary
# ---------------------------

user_dict = {
    row["userId"]: {r["movieId"]: r["rating"] for r in row["ratings"]}
    for row in train.groupBy("userId")
    .agg(collect_list(struct("movieId","rating")).alias("ratings"))
    .collect()
}

# ---------------------------
# 6 Predict Rating
# ---------------------------

def predict_item(user_id, movie_id):

    if user_id not in user_dict:
        return 0

    rated = user_dict[user_id]

    num = 0
    den = 0

    for m, rating in rated.items():

        for (nbr, sim) in item_neighbors.get(movie_id, []):

            if nbr == m:
                num += sim * rating
                den += abs(sim)

    return num/den if den != 0 else 0


# ---------------------------
# 7 Recommend Movies
# ---------------------------

def recommend_item(user_id, N=5):

    rated = user_dict.get(user_id, {})

    scores = {}

    for movie in item_dict.keys():

        if movie in rated:
            continue

        scores[movie] = predict_item(user_id, movie)

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    return ranked[:N]


# ---------------------------
# 8 Ground Truth (Test Set)
# ---------------------------

liked_dict = {
    row["userId"]: row["liked"]
    for row in test.filter(col("rating")>=4)
    .groupBy("userId")
    .agg(collect_list("movieId").alias("liked"))
    .collect()
}

# ---------------------------
# 9 Evaluation Metrics
# ---------------------------

print("\nEvaluating model...")

results=[]
errors=[]

for user in list(liked_dict.keys())[:200]:

    try:

        recs = recommend_item(user,5)
        rec_ids = [r[0] for r in recs]

        relevant = set(liked_dict[user])
        hits = set(rec_ids) & relevant

        precision = len(hits)/5
        recall = len(hits)/len(relevant)

        f1 = (2*precision*recall/(precision+recall)) if precision+recall else 0

        # RMSE
        for movie in relevant:
            pred = predict_item(user, movie)
            if pred != 0:
                errors.append((4 - pred)**2)  # assuming >=4 as liked baseline

        results.append(Row(
            userId=user,
            precision_at_5=float(precision),
            recall_at_5=float(recall),
            f1_score=float(f1)
        ))

    except:
        continue

rmse = sqrt(sum(errors)/len(errors)) if errors else 0

print("RMSE:", rmse)

report_df = spark.createDataFrame(results)
report_df.show(10)

# ---------------------------
# 10 Save CSV
# ---------------------------

output="/home/suvendu/mlbd/item_cf_final"

report_df.coalesce(1).write \
.mode("overwrite") \
.option("header",True) \
.csv(output)

print("\nSaved to:", output)
print("Total time:", time.time()-start)

spark.stop()