# Part 1: Content-Based Filtering
# Task 1: Implementing TF-IDF Based Recommendation
#In this task, you will build a content-based recommender system using Term Frequency-Inverse Document Frequency (TFIDF).
# The goal is to recommend movies similar to a given movie based on textual features.

# Install pyspark if needed
# !pip install pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, lower
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
import os 

# Part 1 — Content Based Filtering using PySpark

'''spark = SparkSession.builder \
    .appName("Movie Recommender System") \
    .getOrCreate()'''
    
spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()    

path ='/home/suvendu/mlbd/'

movies = spark.read.csv(path  + "ml-latest/movies.csv", header=True, inferSchema=True)

print("Movies loaded ------------------- ")
movies.show(5)


# Step 4: Extract Movie Descriptions (Genres)
# Convert genres into tokens.

movies = movies.withColumn("genres", lower(col("genres")))

tokenizer = Tokenizer(inputCol="genres", outputCol="words")

movies_words = tokenizer.transform(movies)


print("extracted the movie descriptions ------------------- ")
movies_words.select("title","words").show(5)


# Step 5: Compute TF (Term Frequency)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)

tf_data = hashingTF.transform(movies_words)


# Step 6: Compute TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")

idf_model = idf.fit(tf_data)

tfidf_data = idf_model.transform(tf_data)

print("TF-IDF vectors using TfidfVectorizer ---------------------- ")

tfidf_data.select("title","features").show(5)


# Step 7: Cosine Similarity Function
def cosine_similarity(v1, v2):
    dot = float(v1.dot(v2))
    normA = float(np.linalg.norm(v1.toArray()))
    normB = float(np.linalg.norm(v2.toArray()))
    
    if normA == 0 or normB == 0:
        return 0.0
    
    return float(dot / (normA * normB))

# Converting it to Spark UDF 
# Wraps the Python function

#Spark cannot directly use normal Python functions inside a DataFrame query.
#So udf() wraps the function so Spark can run it on distributed data.

cosine_udf = udf(cosine_similarity, DoubleType())

# Step 8: Collect Features
movie_features = tfidf_data.select("movieId","title","features").collect()

movie_dict = {row['title']: row['features'] for row in movie_features}

# Step 9: Recommendation Function

def recommend_movies(movie_title, top_n=5):
    
    if movie_title not in movie_dict:
        print("Movie not found!")
        return
    
    query_vector = movie_dict[movie_title]
    
    similarities = []
    
    for title, vector in movie_dict.items():
        if title != movie_title:
            sim = cosine_similarity(query_vector, vector)
            similarities.append((title, sim))
    
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]

# Step 10: Test the Recommender

movie = "Money Train (1995)"  #"Nixon (1995)"
#"Ace Ventura: When Nature Calls (1995)"

recommendations = recommend_movies(movie, 5)

print("Top Recommendations for:", movie)

for title, score in recommendations:
    print(title, " | similarity =", round(score,3))
    

#############################################################################


# Part 1: Content-Based Filtering
# Task 2: User-Profile-Based Content Recommender

# In this task, you will build a content-based recommender system that personalizes recommendations based on user
# preferences. Instead of relying solely on item descriptions, you will construct user profiles from their historical interactions (i.e.,
# rated movies) and use them to suggest relevant movies.


# Install pyspark if needed
# !pip install pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, lower
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import numpy as np
import os 



# We also use ratings.csv to build user preferences.

ratings = spark.read.csv(path + 
    "ml-latest/ratings.csv",
    header=True,
    inferSchema=True
)



print("Movies loaded ------------------- ")
ratings.show(5)


# Step 4: Extract Movie Descriptions Join Ratings with Movie TF-IDF Vectors

user_movie_vectors = ratings.join(
    tfidf_data.select("movieId","features","title"),
    on="movieId"
)

print("extracted the movie descriptions ------------------- ")
user_movie_vectors.show(5)
# | userId | movieId | rating | title | features |




# Step 5: Compute User Profiles

# compute the weighted TF-IDF average.

from pyspark.sql.functions import col
from pyspark.ml.linalg import DenseVector
import numpy as np

# user_data = user_movie_vectors.collect()  # this overflow the memory Java heap space error

from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

from pyspark.sql.functions import expr

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType

#Instead of collecting data, compute the user profile directly in Spark.
# Step 1 — Multiply TF-IDF vector by rating
'''user_movie_vectors = user_movie_vectors.withColumn(
    "weighted_features",
    col("features") * col("rating")
)'''



# Define Prediction Function
def predict_fn(X, user_profile):
    """
    X: movie feature matrix (n_samples, n_features)
    user_profile: user vector
    """
    scores = []

    for vec in X:
        score = cosine_similarity(user_profile, vec)
        scores.append(score)

    return np.array(scores)

# SHAP Explainer

def explain_recommendation_shap(user_id, movie_index):

    user_profile = np.array(user_profiles_dict[user_id])

    # background sample (small subset for speed)
    background = movie_feature_matrix[np.random.choice(len(movie_feature_matrix), 100, replace=False)]

    explainer = shap.KernelExplainer(
        lambda x: predict_fn(x, user_profile),
        background
    )

    # Explain one movie
    movie_vec = movie_feature_matrix[movie_index].reshape(1, -1)

    shap_values = explainer.shap_values(movie_vec)

    return shap_values, movie_vec

# Display Explanation

def print_shap_explanation(user_id, movie_index, top_k=10):

    shap_values, movie_vec = explain_recommendation_shap(user_id, movie_index)

    feature_contrib = shap_values[0]

    # Get top features
    top_features_idx = np.argsort(np.abs(feature_contrib))[::-1][:top_k]

    print("\n🔍 SHAP Explanation")
    print(f"User {user_id} → Movie: {movie_titles[movie_index]}")

    for idx in top_features_idx:
        print(f"Feature {idx}: Contribution {feature_contrib[idx]:.4f}")
        
        
#PySpark, you cannot directly multiply a vector column by a scalar like this:
#Spark SQL does not support vector-scalar multiplication directly.
# must convert the feature vector into an array, multiply each element, and then convert it back
user_movie_vectors = user_movie_vectors.withColumn(
    "features_array",
    vector_to_array("features")
)

# Step 3 — Multiply by Rating 
# Create a UDF that multiplies each element.
def multiply_vector(arr, rating):
    return [x * rating for x in arr]

multiply_udf = udf(multiply_vector, ArrayType(DoubleType()))

# user profile creation

# numaretor 
user_movie_vectors = user_movie_vectors.withColumn(
    "weighted_array",
    multiply_udf(col("features_array"), col("rating"))
)

# Step 4 — Convert Back to Vector

'''from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors

def array_to_vector(arr):
    return Vectors.dense(arr)

array_to_vector_udf = udf(array_to_vector, VectorUDT())

from pyspark.ml.functions import vector_to_array

user_movie_vectors = user_movie_vectors.withColumn(
    "weighted_array",
    vector_to_array("weighted_features")
)'''


# Step 2 — Aggregate per user

from pyspark.sql.functions import sum as spark_sum

rating_sum = user_movie_vectors.groupBy("userId").agg(
    spark_sum("rating").alias("sum_rating")
)

# Step 3 — Aggregate Feature Arrays (UDF)
# Create a function to sum arrays element-wise
'''from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
import numpy as np

def array_sum(arrays):
    return list(np.sum(arrays, axis=0))

array_sum_udf = udf(array_sum, ArrayType(DoubleType()))'''


# Then collect arrays per user:
from pyspark.sql.functions import collect_list

user_vectors = user_movie_vectors.groupBy("userId").agg(
    collect_list("weighted_array").alias("array_list")
)

# Sum Arrays Element-Wise
import numpy as np

def array_sum(arrays):
    return list(np.sum(arrays, axis=0))

array_sum_udf = udf(array_sum, ArrayType(DoubleType()))

user_vectors = user_vectors.withColumn(
    "sum_vector",
    array_sum_udf("array_list")
)

# Join Rating Sum
user_profiles = user_vectors.join(rating_sum, "userId")

# Normalize User Profile

def normalize_vector(vec, rating_sum):
    return [v / rating_sum for v in vec]

normalize_udf = udf(normalize_vector, ArrayType(DoubleType()))

user_profiles = user_profiles.withColumn(
    "user_profile",
    normalize_udf("sum_vector", "sum_rating")
)

print(" Now each user has a profile vector.")
#user_profiles.show(5)

# Now each user has a profile vector.




'''user_profiles = {}

for row in user_data:
    
    user = row['userId']
    rating = row['rating']
    vec = row['features'].toArray()
    
    if user not in user_profiles:
        user_profiles[user] = {
            "vector": rating * vec,
            "weight": rating
        }
    else:
        user_profiles[user]["vector"] += rating * vec
        user_profiles[user]["weight"] += rating
        

# Normalize User Profile
for user in user_profiles:
    
    user_profiles[user]["vector"] = (
        user_profiles[user]["vector"] /
        user_profiles[user]["weight"]
    )'''

#Now each user has a preference vector.

# Computing User-Movie Similarity

# Cosine Similarity Function using User profile vector P𝑢 and Movie TF-IDF vector fm

def cosine_similarity(v1, v2):
    
    dot = np.dot(v1, v2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot / (norm1 * norm2)

# Generate Recommendations for a User

movie_vectors = tfidf_data.select("title","features").collect()

# Convert User Profile to Python Dict

user_profiles_dict = {
    row["userId"]: row["user_profile"]
    for row in user_profiles.select("userId", "user_profile").collect()
}


# Recommendation Function
def recommend_movies_for_user(user_id, top_n=5):
    
    profile = user_profiles[user_id]["vector"]
    
    scores = []
    
    for row in movie_vectors:
        
        movie = row['title']
        vec = row['features'].toArray()
        
        sim = cosine_similarity(profile, vec)
        
        scores.append((movie, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores[:top_n]

import shap

# Convert Spark TF-IDF vectors → NumPy
movie_feature_matrix = np.array([row['features'].toArray() for row in movie_vectors])
movie_titles = [row['title'] for row in movie_vectors]

# Test User Recommendation
recommend_movies_for_user(1,5)

# Example: explain top recommendation
user_id = 1
recs = recommend_movies_for_user(user_id, 5)

# pick first movie
movie_name = recs[0][0]

movie_index = movie_titles.index(movie_name)

print_shap_explanation(user_id, movie_index)

vocab = cv_model.vocabulary
print(f"{vocab[idx]}: {contribution}")

spark.stop()
         