# Part 2: Collaborative Filtering
# Task 3: User-Based Collaborative Filtering

# This code is showing out of memory issue --- so the modified one is "pyspark_assignment3_2_v1.py"

# ============================================
# USER BASED COLLABORATIVE FILTERING
# DEBUG VERSION
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct
import numpy as np
from pyspark.sql import Row
import time

start_total=time.time()

print("\nStarting User-Based Collaborative Filtering")

spark = SparkSession.builder \
    .appName("UserCF") \
    .config("spark.driver.memory","6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

path="/home/suvendu/mlbd/ml-latest-small/"

# ---------------------------
# 1 Load Data
# ---------------------------

print("\nStep 1: Loading Dataset...")

t=time.time()

movies=spark.read.csv(
    path+"movies.csv",
    header=True,
    inferSchema=True
)

ratings=spark.read.csv(
    path+"ratings.csv",
    header=True,
    inferSchema=True
)

print("Movies count:",movies.count())
print("Ratings count:",ratings.count())

print("Load time:",time.time()-t,"seconds")


# ---------------------------
# 2 Build User Rating Matrix
# ---------------------------

print("\nStep 2: Building User Rating Matrix")

t=time.time()

user_ratings=ratings.groupBy("userId").agg(
    collect_list(struct("movieId","rating")).alias("ratings")
)

print("User groups created")

user_dict={
    row["userId"]:
    {r["movieId"]:r["rating"] for r in row["ratings"]}
    for row in user_ratings.collect()
}

users=list(user_dict.keys())

print("Total users:",len(users))
print("Matrix build time:",time.time()-t,"seconds")


# ---------------------------
# 3 Similarity Function
# ---------------------------

print("\nStep 3: Similarity function ready")


def pearson_similarity(u1,u2):

    common=set(u1.keys()) & set(u2.keys())

    if len(common)==0:
        return 0

    r1=np.array([u1[m] for m in common])
    r2=np.array([u2[m] for m in common])

    mean1=np.mean(r1)
    mean2=np.mean(r2)

    num=np.sum((r1-mean1)*(r2-mean2))
    den=np.sqrt(np.sum((r1-mean1)**2))*np.sqrt(np.sum((r2-mean2)**2))

    if den==0:
        return 0

    return num/den


# ---------------------------
# 4 Compute User Similarity
# ---------------------------

print("\nStep 4: Computing User Similarity")

t=time.time()

user_similarity={}

for i in range(len(users)):

    if i%50==0:
        print("Processing user similarity:",i,"/",len(users))

    u1=users[i]
    user_similarity[u1]={}

    for j in range(i+1,len(users)):

        u2=users[j]

        sim=pearson_similarity(
            user_dict[u1],
            user_dict[u2]
        )

        if sim>0:
            user_similarity[u1][u2]=sim

print("User similarity computation finished")
print("Similarity time:",time.time()-t,"seconds")


# ---------------------------
# 5 Predict Rating
# ---------------------------

print("\nStep 5: Rating Prediction Function Ready")

def predict_rating(user_id,movie_id,K=5):

    neighbors=[]

    for other in users:

        if other==user_id:
            continue

        if movie_id in user_dict.get(other,{}):
            sim=pearson_similarity(
                user_dict[user_id],
                user_dict[other]
            )

            neighbors.append(
                (other,sim,user_dict[other][movie_id])
            )

    neighbors=sorted(
        neighbors,
        key=lambda x:x[1],
        reverse=True
    )[:K]

    num=0
    den=0

    for n in neighbors:

        sim=n[1]
        rating=n[2]

        num+=sim*rating
        den+=abs(sim)

    if den==0:
        return 0

    return num/den


# ---------------------------
# 6 Recommendation Function
# ---------------------------

print("\nStep 6: Recommendation Function Ready")

movie_ids=[row.movieId for row in movies.select("movieId").collect()]

def recommend_movies(user_id,N=5,K=5):

    rated=user_dict[user_id]

    scores={}

    for movie in movie_ids:

        if movie in rated:
            continue

        pred=predict_rating(user_id,movie,K)

        scores[movie]=pred

    ranked=sorted(
        scores.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return ranked[:N]


# ---------------------------
# 7 Ground Truth
# ---------------------------

print("\nStep 7: Preparing Ground Truth")

t=time.time()

liked=ratings.filter(col("rating")>=4)

liked_rows=liked.groupBy("userId").agg(
    collect_list("movieId").alias("liked")
).collect()

liked_dict={
    row["userId"]:row["liked"]
    for row in liked_rows
}

print("Ground truth users:",len(liked_dict))
print("Ground truth time:",time.time()-t)


# ---------------------------
# 8 Movie Title Mapping
# ---------------------------

print("\nStep 8: Loading Movie Titles")

movie_map=movies.select("movieId","title").collect()

id_title={
    row["movieId"]:row["title"]
    for row in movie_map
}

print("Movie mapping ready")


# ---------------------------
# 9 Evaluation
# ---------------------------

print("\nStep 9: Starting Evaluation")

t=time.time()

results=[]
K_values=[3,5,10]

for K in K_values:

    print("\nEvaluating for K =",K)

    for user in liked_dict.keys():

        if user%50==0:
            print("Evaluating user:",user)

        try:

            recs=recommend_movies(user,5,K)

            rec_ids=[r[0] for r in recs]

            relevant=set(liked_dict[user])

            hits=set(rec_ids).intersection(relevant)

            precision=len(hits)/5
            recall=len(hits)/len(relevant)

            if precision+recall==0:
                f1=0
            else:
                f1=2*precision*recall/(precision+recall)

            rec_titles=[
                id_title[m] for m in rec_ids
                if m in id_title
            ]

            liked_titles=[
                id_title[m] for m in relevant
                if m in id_title
            ]

            results.append(Row(
                userId=user,
                neighbors_K=K,
                precision_at_5=float(precision),
                recall_at_5=float(recall),
                f1_score=float(f1),
                recommended_movies=", ".join(rec_titles),
                liked_movies=", ".join(liked_titles)
            ))

        except:
            continue

print("Evaluation time:",time.time()-t)


# ---------------------------
# 10 Save CSV
# ---------------------------

print("\nStep 10: Saving CSV")

output="/home/suvendu/mlbd/user_cf_report"

report_df=spark.createDataFrame(results)

report_df.coalesce(1).write \
.mode("overwrite") \
.option("header",True) \
.csv(output)

print("CSV saved at:",output)

print("\nTotal runtime:",time.time()-start_total,"seconds")

spark.stop()