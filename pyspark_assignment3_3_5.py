# ============================================
# SVD BASED RECOMMENDER SYSTEM
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import numpy as np
from math import sqrt
import time

spark = SparkSession.builder \
    .appName("SVD_Recommender") \
    .config("spark.driver.memory","6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

path="/home/suvendu/mlbd/ml-latest-small/"

start=time.time()

# ---------------------------
# 1 Load Data + Split
# ---------------------------

print("\nLoading data...")

ratings = spark.read.csv(path+"ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv(path+"movies.csv", header=True, inferSchema=True)

train, test = ratings.randomSplit([0.8,0.2], seed=42)

# ---------------------------
# 2 Create User-Item Matrix
# ---------------------------

print("\nBuilding user-item matrix...")

users = [row.userId for row in train.select("userId").distinct().collect()]
items = [row.movieId for row in train.select("movieId").distinct().collect()]

user_index = {u:i for i,u in enumerate(users)}
item_index = {m:i for i,m in enumerate(items)}

R = np.zeros((len(users), len(items)))

for row in train.collect():
    u = user_index[row["userId"]]
    m = item_index[row["movieId"]]
    R[u][m] = row["rating"]

print("Matrix shape:", R.shape)

# ---------------------------
# 3 Normalize (IMPORTANT)
# ---------------------------

print("\nNormalizing matrix...")

user_means = np.mean(R, axis=1)

R_norm = R.copy()

for i in range(len(users)):
    for j in range(len(items)):
        if R[i][j] != 0:
            R_norm[i][j] -= user_means[i]

# ---------------------------
# 4 Apply SVD
# ---------------------------

print("\nApplying SVD...")

U, sigma, Vt = np.linalg.svd(R_norm, full_matrices=False)

# reduce dimensions
k = 20

U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# ---------------------------
# 5 Reconstruct Ratings
# ---------------------------

print("\nReconstructing matrix...")

R_pred = np.dot(np.dot(U_k, sigma_k), Vt_k)

# add mean back
for i in range(len(users)):
    R_pred[i] += user_means[i]

# ---------------------------
# 6 Recommendation Function
# ---------------------------

def recommend_svd(user_id, N=5):

    if user_id not in user_index:
        return []

    u = user_index[user_id]

    scores = list(enumerate(R_pred[u]))

    rated_movies = set(
        [row.movieId for row in train.filter(col("userId")==user_id).collect()]
    )

    recs = [
        (items[i], score)
        for i, score in scores
        if items[i] not in rated_movies
    ]

    recs = sorted(recs, key=lambda x:x[1], reverse=True)

    return recs[:N]

# ---------------------------
# 7 Ground Truth
# ---------------------------

liked_dict = {
    row["userId"]: row["liked"]
    for row in test.filter(col("rating")>=4)
    .groupBy("userId")
    .agg({"movieId":"collect_list"})
    .withColumnRenamed("collect_list(movieId)", "liked")
    .collect()
}

# ---------------------------
# 8 Evaluation
# ---------------------------

print("\nEvaluating SVD...")

precision_list=[]
recall_list=[]
errors=[]

for user in list(liked_dict.keys())[:200]:

    if user not in user_index:
        continue

    recs = recommend_svd(user,5)
    rec_ids = [r[0] for r in recs]

    relevant = set(liked_dict[user])
    hits = set(rec_ids) & relevant

    precision = len(hits)/5
    recall = len(hits)/len(relevant)

    precision_list.append(precision)
    recall_list.append(recall)

    # RMSE
    for movie in relevant:
        if movie in item_index:
            pred = R_pred[user_index[user]][item_index[movie]]
            errors.append((4 - pred)**2)

# metrics
precision_avg = sum(precision_list)/len(precision_list)
recall_avg = sum(recall_list)/len(recall_list)
rmse = sqrt(sum(errors)/len(errors)) if errors else 0

print("\nResults:")
print("Precision@5:", precision_avg)
print("Recall@5:", recall_avg)
print("RMSE:", rmse)

print("\nTotal time:", time.time()-start)

spark.stop()