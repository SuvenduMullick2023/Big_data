# Part 2: Collaborative Filtering
# Task 3: User-Based Collaborative Filtering
# ============================================
# FAST USER BASED COLLABORATIVE FILTERING
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct
import numpy as np
from pyspark.sql import Row
import time

spark = SparkSession.builder \
    .appName("FastUserCF") \
    .config("spark.driver.memory","6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

path="/home/suvendu/mlbd/ml-latest-small/"

start=time.time()

# ---------------------------
# 1 Load Data
# ---------------------------

print("\nLoading data...")

movies = spark.read.csv(path+"movies.csv", header=True, inferSchema=True)
all_ratings = spark.read.csv(path+"ratings.csv", header=True, inferSchema=True)

ratings, test = all_ratings.randomSplit([0.8, 0.2], seed=42)

print("Train:", ratings.count())
print("Test:", test.count())

# ---------------------------
# 2 Build User Dictionary
# ---------------------------

print("\nBuilding user rating matrix...")

user_dict = {
    row["userId"]: {r["movieId"]: r["rating"] for r in row["ratings"]}
    for row in ratings.groupBy("userId")
    .agg(collect_list(struct("movieId","rating")).alias("ratings"))
    .collect()
}

users = list(user_dict.keys())

print("Users:", len(users))


# ---------------------------
# 3 Pearson Similarity
# ---------------------------

def pearson(u1,u2):

    common = set(u1.keys()) & set(u2.keys())

    if len(common) < 1:   #  pruning (important!)
        return 0

    r1 = np.array([u1[m] for m in common])
    r2 = np.array([u2[m] for m in common])

    r1_mean = r1.mean()
    r2_mean = r2.mean()

    num = np.sum((r1 - r1_mean)*(r2 - r2_mean))
    den = np.sqrt(np.sum((r1 - r1_mean)**2)) * np.sqrt(np.sum((r2 - r2_mean)**2))

    return num/den if den != 0 else 0


# ---------------------------
# 4 Precompute Top-K Neighbors
# ---------------------------

print("\nComputing Top-K neighbors...")

K_SIM = 20   # neighbors to store

user_neighbors = {}

for i,u1 in enumerate(users):

    if i % 50 == 0:
        print("Processing user:", i)

    sims = []

    for u2 in users:

        if u1 == u2:
            continue

        sim = pearson(user_dict[u1], user_dict[u2])

        if sim > 0:
            sims.append((u2, sim))

    #  keep only top K similar users
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:K_SIM]

    user_neighbors[u1] = sims

print("Neighbor computation done")


# ---------------------------
# 5 Predict Ratings (FAST)
# ---------------------------

def predict_fast(user_id, movie_id):

    neighbors = user_neighbors.get(user_id, [])
    user_mean = np.mean(list(user_dict[user_id].values()))

    num, den = 0, 0

    for (nbr, sim) in neighbors:
        if movie_id in user_dict[nbr]:
            nbr_mean = np.mean(list(user_dict[nbr].values()))
            num += sim * (user_dict[nbr][movie_id] - nbr_mean)
            den += abs(sim)

    if den == 0:
        pred = user_mean
    else:
        pred = user_mean + (num/den)

    #  CLAMP
    return min(5, max(0.5, pred))

#  Task 11: Neighborhood-Based Explanations without breaking  logic.
# ---------------------------
#  EXPLANATION FUNCTION
# ---------------------------
def explain_recommendation(user_id, movie_id, top_k=3):

    neighbors = user_neighbors.get(user_id, [])

    explanation = []

    for nbr, sim in neighbors:
        if movie_id in user_dict[nbr]:
            explanation.append((nbr, sim, user_dict[nbr][movie_id]))

    # Sort by similarity
    explanation = sorted(explanation, key=lambda x: x[1], reverse=True)[:top_k]

    return explanation



# ---------------------------
# 6 Recommend Movies (FAST)
# ---------------------------

def recommend_fast_old(user_id, N=5):

    rated = user_dict[user_id]

    candidate_movies = set()

    #  only consider movies rated by neighbors
    for nbr,_ in user_neighbors[user_id]:
        candidate_movies.update(user_dict[nbr].keys())

    scores = {}

    for movie in candidate_movies:

        if movie in rated:
            continue

        scores[movie] = predict_fast(user_id, movie)

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    return ranked[:N]

# Task 11: Neighborhood-Based Explanations without breaking  logic. upgradation 
def recommend_fast(user_id, N=5):

    rated = user_dict[user_id]

    candidate_movies = set()

    for nbr,_ in user_neighbors[user_id]:
        candidate_movies.update(user_dict[nbr].keys())

    scores = {}
    explanations = {}

    for movie in candidate_movies:

        if movie in rated:
            continue

        pred = predict_fast(user_id, movie)
        scores[movie] = pred

        #  store explanation
        explanations[movie] = explain_recommendation(user_id, movie)

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    # return movie + explanation
    return [(m, score, explanations[m]) for m, score in ranked[:N]]


# ---------------------------
# 7 Ground Truth
# ---------------------------

'''liked_dict = {
    row["userId"]: row["liked"]
    for row in ratings.filter(col("rating")>=4)
    .groupBy("userId")
    .agg(collect_list("movieId").alias("liked"))
    .collect()
}'''

liked_dict = {
    row["userId"]: row["liked"]
    for row in test.filter(col("rating")>=4)
    .groupBy("userId")
    .agg(collect_list("movieId").alias("liked"))
    .collect()
}


# ---------------------------
# 8 Movie Mapping
# ---------------------------

id_title = {
    row["movieId"]: row["title"]
    for row in movies.select("movieId","title").collect()
}


# ---------------------------
# 9 Evaluation
# ---------------------------

print("\nEvaluating model...")

results=[]

'''for user in list(liked_dict.keys())[:200]:   # 🔥 limit for speed

    try:

        recs = recommend_fast(user,5)

        rec_ids = [r[0] for r in recs]

        relevant = set(liked_dict[user])

        hits = set(rec_ids) & relevant

        precision = len(hits)/5
        recall = len(hits)/len(relevant)

        f1 = (2*precision*recall/(precision+recall)) if precision+recall else 0

        results.append(Row(
            userId=user,
            precision_at_5=float(precision),
            recall_at_5=float(recall),
            f1_score=float(f1)
        ))

    except:
        continue'''

# Task 11: Neighborhood-Based Explanations without breaking  logic. upgradation 
for user in list(liked_dict.keys())[:10]:   # reduce for readability

    try:
        recs = recommend_fast(user,5)

        print("\n==============================")
        print(f"User {user} Recommendations")

        rec_ids = []

        for movie, score, expl in recs:

            rec_ids.append(movie)

            print(f"\n🎬 {id_title.get(movie, movie)} (Pred: {round(score,2)})")

            #  Explanation
            print("    Because similar users liked it:")

            for nbr, sim, rating in expl:
                print(f"      User {nbr} (sim={round(sim,2)}) rated {rating}")

        relevant = set(liked_dict[user])
        hits = set(rec_ids) & relevant

        precision = len(hits)/5
        recall = len(hits)/len(relevant)

        results.append(Row(
            userId=user,
            precision_at_5=float(precision),
            recall_at_5=float(recall),
            f1_score=float((2*precision*recall/(precision+recall)) if precision+recall else 0)
        ))

    except:
        continue

# Add Item-Based Explanation
# ----------------------------------------------------------
def explain_item_based(movie_id, top_k=3):

    users_who_liked = [
        u for u in user_dict
        if movie_id in user_dict[u] and user_dict[u][movie_id] >= 4
    ]

    similar_movies = {}

    for u in users_who_liked:
        for m in user_dict[u]:
            if m != movie_id:
                similar_movies[m] = similar_movies.get(m, 0) + 1

    ranked = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]

# ----------------------------------------------------------------------

report_df = spark.createDataFrame(results)

report_df.show(10)

from math import sqrt

def compute_rmse():

    errors = []

    for row in test.collect():

        user = row["userId"]
        movie = row["movieId"]
        true = row["rating"]

        if user in user_dict:
            pred = predict_fast(user, movie)

            if pred != 0:
                errors.append((true - pred)**2)

    rmse = sqrt(sum(errors)/len(errors)) if errors else 0

    print("RMSE:", rmse)

compute_rmse()
# ---------------------------
# 10 Save CSV
# ---------------------------

output="/home/suvendu/mlbd/fast_user_cf"

report_df.coalesce(1).write \
.mode("overwrite") \
.option("header",True) \
.csv(output)

print("\nSaved to:",output)

print("User:", user)
print("Recommended:", rec_ids)
print("Relevant:", list(relevant)[:10])
print("Hits:", hits)

print("Total Time:", time.time()-start)

spark.stop()