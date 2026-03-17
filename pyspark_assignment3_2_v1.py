# ==============================
# Movie Recommendation System
# Content + User Profile based
# PySpark Implementation
# ==============================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import numpy as np

# ------------------------------
# 1 Create Spark Session
# ------------------------------

spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .getOrCreate()

path = "/home/suvendu/mlbd/"

# ------------------------------
# 2 Load Movies Dataset
# ------------------------------

movies = spark.read.csv(
    path + "ml-latest-small/movies.csv",
    header=True,
    inferSchema=True
)

print("Movies Loaded")
movies.show(5)

# ------------------------------
# 3 Extract Movie Genres
# ------------------------------

movies = movies.withColumn("genres", lower(col("genres")))

tokenizer = Tokenizer(
    inputCol="genres",
    outputCol="words"
)

movies_words = tokenizer.transform(movies)

print("Tokenized Genres")
movies_words.select("title","words").show(5)

# ------------------------------
# 4 Compute TF
# ------------------------------

hashingTF = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=1000
)

tf_data = hashingTF.transform(movies_words)

# ------------------------------
# 5 Compute TF-IDF
# ------------------------------

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

idf_model = idf.fit(tf_data)

tfidf_data = idf_model.transform(tf_data)

tfidf_data = tfidf_data.select("movieId","title","features").cache()

print("TF-IDF Vectors Created")
tfidf_data.show(5)

# ------------------------------
# 6 Load Ratings Dataset
# ------------------------------

ratings = spark.read.csv(
    path + "ml-latest-small/ratings.csv",
    header=True,
    inferSchema=True
)

print("Ratings Loaded")
ratings.show(5)

# ------------------------------
# 7 Join Ratings with TF-IDF
# ------------------------------

user_movie_vectors = ratings.join(
    tfidf_data,
    on="movieId"
)

# Convert vector to array
user_movie_vectors = user_movie_vectors.withColumn(
    "features_array",
    vector_to_array("features")
)

# ------------------------------
# 8 Multiply TF-IDF by Rating
# ------------------------------

def multiply_vector(arr, rating):
    return [float(x * rating) for x in arr]

multiply_udf = udf(multiply_vector, ArrayType(DoubleType()))

user_movie_vectors = user_movie_vectors.withColumn(
    "weighted_array",
    multiply_udf(col("features_array"), col("rating"))
)

print("Weighted TF-IDF Computed")

# ------------------------------
# 9 Sum Ratings per User
# ------------------------------

from pyspark.sql.functions import sum as spark_sum

rating_sum = user_movie_vectors.groupBy("userId").agg(
    spark_sum("rating").alias("sum_rating")
)

# ------------------------------
# 10 Sum Weighted Vectors
# ------------------------------

from pyspark.sql.functions import posexplode

exploded = user_movie_vectors.select(
    "userId",
    posexplode("weighted_array").alias("index","value")
)

feature_sum = exploded.groupBy("userId","index").agg(
    spark_sum("value").alias("sum_value")
)

# ------------------------------
# 11 Normalize User Profile
# ------------------------------

user_profile = feature_sum.join(rating_sum,"userId")

user_profile = user_profile.withColumn(
    "normalized",
    col("sum_value") / col("sum_rating")
)

# ------------------------------
# 12 Rebuild Profile Vector
# ------------------------------

from pyspark.sql.functions import collect_list, struct, sort_array

user_profile = user_profile.groupBy("userId").agg(
    sort_array(
        collect_list(struct("index","normalized"))
    ).alias("profile")
)

def extract_values(arr):
    return [x["normalized"] for x in arr]

extract_udf = udf(extract_values, ArrayType(DoubleType()))

user_profile = user_profile.withColumn(
    "user_profile",
    extract_udf("profile")
).select("userId","user_profile")

print("User Profiles Created")

# ------------------------------
# 13 Convert Profiles to Python
# ------------------------------

user_profiles_dict = {
    row["userId"]: np.array(row["user_profile"])
    for row in user_profile.collect()
}

movie_vectors = tfidf_data.select("title","features").collect()

print("Converted Profiles to Dictionary")

# ------------------------------
# 14 Cosine Similarity
# ------------------------------

def cosine_similarity(v1, v2):

    dot = np.dot(v1, v2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)

# ------------------------------
# 15 Recommendation Function
# ------------------------------

def recommend_movies_for_user(user_id, top_n=5):

    if user_id not in user_profiles_dict:
        print("User not found")
        return

    profile = user_profiles_dict[user_id]

    scores = []

    for row in movie_vectors:

        movie = row["title"]
        vec = row["features"].toArray()

        sim = cosine_similarity(profile, vec)

        scores.append((movie, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:top_n]

# ------------------------------
# 16 Test Recommendation
# ------------------------------

user_id = 1
for user_id in range(1,5):
    
    recommendations = recommend_movies_for_user(user_id,5)

    print("\nTop Recommendations for User", user_id)

    for movie, score in recommendations:
        print(movie, "| similarity =", round(score,3))
        

# Ground truth: movies user liked
liked_movies = ratings.filter(col("rating") >= 4)

liked_movies.show(5)

# Build User → Liked Movie Dictionary
liked_dict = (
    liked_movies.groupBy("userId")
    .agg(collect_list("movieId").alias("liked_movies"))
    .collect()
)

liked_dict = {row["userId"]: row["liked_movies"] for row in liked_dict}  

# Map Movie Title → MovieId 
# recommender returns titles, but ratings use movieId.
# So build a mapping.

movie_map = tfidf_data.select("movieId","title").collect()

title_to_id = {row["title"]: row["movieId"] for row in movie_map}

# Precision@K and Recall@K Function

def evaluate_user(user_id, K=5):

    if user_id not in liked_dict:
        print("User has no liked movies")
        return

    # Ground truth
    relevant_movies = set(liked_dict[user_id])

    # Recommendations
    recs = recommend_movies_for_user(user_id, K)

    rec_movie_ids = []

    for title, score in recs:
        if title in title_to_id:
            rec_movie_ids.append(title_to_id[title])

    rec_movie_ids = set(rec_movie_ids)

    # Intersection
    hits = rec_movie_ids.intersection(relevant_movies)

    precision = len(hits) / K

    recall = len(hits) / len(relevant_movies)

    print("User:", user_id)
    print("Hits:", hits)

    print("Precision@",K,"=",round(precision,3))
    print("Recall@",K,"=",round(recall,3))

    return precision, recall


# Run Evaluation
evaluate_user(1,5) 

# Evaluate Over All Users (Recommended)  
# Better evaluation is average metrics across users.
def evaluate_all_users(K=5):

    precisions = []
    recalls = []

    for user in liked_dict.keys():

        try:
            p,r = evaluate_user(user,K)
            precisions.append(p)
            recalls.append(r)
        except:
            continue

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)

    print("\nAverage Precision@",K,"=",round(avg_precision,3))
    print("Average Recall@",K,"=",round(avg_recall,3))
    
     
evaluate_all_users(5)

from pyspark.sql import Row

results = []

K = 5

for user in liked_dict.keys():
    
    try:
        # Ground truth
        relevant_movies = set(liked_dict[user])
        
        # Recommendations
        recs = recommend_movies_for_user(user, K)
        
        rec_movie_ids = []
        
        for title, score in recs:
            if title in title_to_id:
                rec_movie_ids.append(title_to_id[title])
        
        rec_movie_ids = set(rec_movie_ids)
        
        # Intersection
        hits = rec_movie_ids.intersection(relevant_movies)
        
        precision = len(hits) / K
        recall = len(hits) / len(relevant_movies)
        
        results.append(Row(
            userId=user,
            precision_at_k=float(precision),
            recall_at_k=float(recall),
            hits=len(hits)
        ))
        
    except:
        continue
    
# Convert Results to Spark DataFrame
evaluation_df = spark.createDataFrame(results)

evaluation_df.show(10)
output_path = path + "evaluation_results"

evaluation_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)
    


# Build Movie Title Mapping
from pyspark.sql.functions import collect_list

movie_map = movies.select("movieId","title").collect()

id_to_title = {row["movieId"]: row["title"] for row in movie_map}
title_to_id = {row["title"]: row["movieId"] for row in movie_map}

# Get User Liked Movies (Ground Truth)
liked_movies = ratings.filter(col("rating") >= 4)

liked_dict_rows = liked_movies.groupBy("userId").agg(
    collect_list("movieId").alias("liked_movies")
).collect()

liked_dict = {
    row["userId"]: row["liked_movies"]
    for row in liked_dict_rows
}

# Evaluation + Recommendation Report
from pyspark.sql import Row

K = 5
report_rows = []

for user in liked_dict.keys():

    try:

        # Ground truth liked movies
        relevant_movies = set(liked_dict[user])

        liked_titles = [
            id_to_title[m] for m in relevant_movies if m in id_to_title
        ]

        # Get recommendations
        recs = recommend_movies_for_user(user, K)

        rec_titles = []
        rec_ids = []

        for title, score in recs:

            rec_titles.append(title)

            if title in title_to_id:
                rec_ids.append(title_to_id[title])

        rec_ids = set(rec_ids)

        # Compute hits
        hits = rec_ids.intersection(relevant_movies)

        precision = len(hits) / K
        recall = len(hits) / len(relevant_movies)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        report_rows.append(Row(
            userId=user,
            precision_at_5=float(precision),
            recall_at_5=float(recall),
            f1_score=float(f1),
            recommended_movies=", ".join(rec_titles),
            liked_movies=", ".join(liked_titles)
        ))

    except:
        continue
    
# Convert to Spark DataFrame
report_df = spark.createDataFrame(report_rows)

report_df.show(10, truncate=False)

output_path = path + "recommender_evaluation_report"

report_df.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)
    
# Convert to Single CSV File
import glob
import shutil

csv_file = glob.glob(output_path + "/part*.csv")[0]

shutil.move(csv_file, path + "recommender_evaluation_report.csv")

shutil.rmtree(output_path)

                

# ------------------------------
# 17 Stop Spark
# ------------------------------

spark.stop()