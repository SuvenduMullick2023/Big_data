# ============================================
# HYBRID RECOMMENDER (META-LEARNING)
#  Task 7: Implementing a Hybrid Recommendation Model
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

spark = SparkSession.builder.appName("Hybrid_Model").getOrCreate()

# ---------------------------
# 1 Load Data
# ---------------------------

ratings = spark.read.csv("/home/suvendu/mlbd/ml-latest-small/ratings.csv",
                         header=True, inferSchema=True)

movies = spark.read.csv("/home/suvendu/mlbd/ml-latest-small/movies.csv",
                        header=True, inferSchema=True)

train, test = ratings.randomSplit([0.8,0.2], seed=42)

# ---------------------------
# 2 CF Predictions (Surprise SVD style)
# ---------------------------

# Convert to pandas for simplicity
train_pd = train.toPandas()
test_pd = test.toPandas()

# user & movie mean
user_mean = train_pd.groupby("userId")["rating"].mean().to_dict()
movie_mean = train_pd.groupby("movieId")["rating"].mean().to_dict()

global_mean = train_pd["rating"].mean()

def cf_predict(u, m):
    return user_mean.get(u, global_mean) + movie_mean.get(m, global_mean) - global_mean

# ---------------------------
# 3 Content-Based Filtering (CBF)
# ---------------------------

# Simple genre-based similarity
movies_pd = movies.toPandas()

movies_pd['genres'] = movies_pd['genres'].str.split('|')

# build genre vector
all_genres = list(set(g for genres in movies_pd['genres'] for g in genres))

genre_index = {g:i for i,g in enumerate(all_genres)}

def genre_vector(genres):
    vec = np.zeros(len(all_genres))
    for g in genres:
        if g in genre_index:
            vec[genre_index[g]] = 1
    return vec

movies_pd['vec'] = movies_pd['genres'].apply(genre_vector)

movie_vec = dict(zip(movies_pd['movieId'], movies_pd['vec']))

# user profile
user_profiles = {}

for u in train_pd['userId'].unique():
    user_movies = train_pd[train_pd['userId']==u]

    vecs = []
    for _, row in user_movies.iterrows():
        if row['movieId'] in movie_vec:
            vecs.append(movie_vec[row['movieId']] * row['rating'])

    if vecs:
        user_profiles[u] = np.mean(vecs, axis=0)

def cbf_predict(u, m):
    if u not in user_profiles or m not in movie_vec:
        return global_mean

    sim = np.dot(user_profiles[u], movie_vec[m])
    return sim

# ---------------------------
# 4 Build Training Data for Meta Model
# ---------------------------

rows = []

for _, row in train_pd.iterrows():

    u = row['userId']
    m = row['movieId']
    actual = row['rating']

    cf = cf_predict(u,m)
    cbf = cbf_predict(u,m)

    popularity = movie_mean.get(m, global_mean)
    user_bias = user_mean.get(u, global_mean)

    rows.append([cf, cbf, popularity, user_bias, actual])

df_meta = pd.DataFrame(rows, columns=[
    'cf','cbf','popularity','user_bias','rating'
])

# ---------------------------
# 5 Train Meta Model
# ---------------------------

X = df_meta[['cf','cbf','popularity','user_bias']]
y = df_meta['rating']

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X,y)

# ---------------------------
# 6 Prediction Function
# ---------------------------

def hybrid_predict(u, m):

    cf = cf_predict(u,m)
    cbf = cbf_predict(u,m)

    popularity = movie_mean.get(m, global_mean)
    user_bias = user_mean.get(u, global_mean)

    features = pd.DataFrame([{
        'cf': cf,
        'cbf': cbf,
        'popularity': popularity,
        'user_bias': user_bias
    }])

    return model.predict(features)[0]

# ---------------------------
# 7 Evaluation
# ---------------------------

errors=[]
precision_list=[]
recall_list=[]

threshold=4

test_users = test_pd['userId'].unique()[:200]

for u in test_users:

    user_data = test_pd[test_pd['userId']==u]

    preds=[]
    actuals=[]

    for _, row in user_data.iterrows():
        m = row['movieId']
        pred = hybrid_predict(u,m)

        preds.append((m,pred))
        actuals.append((m,row['rating']))

        errors.append((row['rating'] - pred)**2)

    # top-5
    preds_sorted = sorted(preds, key=lambda x:x[1], reverse=True)[:5]
    rec_ids = [p[0] for p in preds_sorted]

    relevant = [m for m,r in actuals if r>=threshold]

    hits = set(rec_ids) & set(relevant)

    precision = len(hits)/5 if len(rec_ids)>0 else 0
    recall = len(hits)/len(relevant) if len(relevant)>0 else 0

    precision_list.append(precision)
    recall_list.append(recall)

rmse = sqrt(sum(errors)/len(errors))
precision_avg = np.mean(precision_list)
recall_avg = np.mean(recall_list)

print("\nHYBRID RESULTS")
print("RMSE:", rmse)
print("Precision@5:", precision_avg)
print("Recall@5:", recall_avg)



# ============================================
# COLD-START USER ANALYSIS
# ============================================

print("\n--- Cold-Start User Analysis ---")

# ---------------------------
# 1 Identify Cold-Start Users
# ---------------------------

# Users with <= 3 ratings in training
user_counts = train_pd.groupby("userId").size()
print(user_counts)
cold_users = user_counts[user_counts <= 25].index.tolist()

print("Number of cold-start users:", len(cold_users))

# ---------------------------
# 2 Filter Test Data
# ---------------------------

cold_test = test_pd[test_pd['userId'].isin(cold_users)]

print("Cold-start test samples:", len(cold_test))

# ---------------------------
# 3 Evaluation Function
# ---------------------------

def evaluate_model(predict_func, name):

    errors=[]
    precision_list=[]
    recall_list=[]

    threshold=4

    test_users = cold_test['userId'].unique()

    for u in test_users:

        user_data = cold_test[cold_test['userId']==u]

        preds=[]
        actuals=[]

        for _, row in user_data.iterrows():
            m = row['movieId']
            true_r = row['rating']

            pred = predict_func(u,m)

            preds.append((m,pred))
            actuals.append((m,true_r))

            errors.append((true_r - pred)**2)

        # top-5 recommendations
        preds_sorted = sorted(preds, key=lambda x:x[1], reverse=True)[:5]
        rec_ids = [p[0] for p in preds_sorted]

        relevant = [m for m,r in actuals if r>=threshold]

        hits = set(rec_ids) & set(relevant)

        precision = len(hits)/5 if len(rec_ids)>0 else 0
        recall = len(hits)/len(relevant) if len(relevant)>0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    rmse = sqrt(sum(errors)/len(errors)) if errors else 0
    precision_avg = np.mean(precision_list) if precision_list else 0
    recall_avg = np.mean(recall_list) if recall_list else 0

    print(f"\n{name} Cold-Start Results")
    print("RMSE:", rmse)
    print("Precision@5:", precision_avg)
    print("Recall@5:", recall_avg)

# ---------------------------
# 4 Define Model Wrappers
# ---------------------------

# Hybrid
def hybrid_wrapper(u,m):
    return hybrid_predict(u,m)

# CF (baseline)
def cf_wrapper(u,m):
    return cf_predict(u,m)

# CBF
def cbf_wrapper(u,m):
    return cbf_predict(u,m)

# ---------------------------
# 5 Run Evaluation
# ---------------------------

evaluate_model(cf_wrapper, "CF Model")
evaluate_model(cbf_wrapper, "CBF Model")
evaluate_model(hybrid_wrapper, "Hybrid Model")