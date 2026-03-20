# Part 3: Matrix Factorization for Recommender Systems
# Task 6: Implementing Matrix Factorization with the Surprise Library
# ============================================
# SURPRISE SVD RECOMMENDER SYSTEM
# ============================================

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
from collections import defaultdict
import pandas as pd
import numpy as np

# ---------------------------
# 1 Load Dataset
# ---------------------------

print("\nLoading dataset...")

ratings_df = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings_df[['userId','movieId','rating']],
    reader
)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

print("Train/Test split done")

# ---------------------------
# 2 Hyperparameter Tuning
# ---------------------------

print("\nTuning SVD...")

param_grid = {
    'n_factors': [20, 50],
    'n_epochs': [10, 20],
    'lr_all': [0.005, 0.01],
    'reg_all': [0.02, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)

gs.fit(data)

print("Best RMSE:", gs.best_score['rmse'])
print("Best Params:", gs.best_params['rmse'])

# ---------------------------
# 3 Train Final Model
# ---------------------------

best_params = gs.best_params['rmse']

model = SVD(
    n_factors=best_params['n_factors'],
    n_epochs=best_params['n_epochs'],
    lr_all=best_params['lr_all'],
    reg_all=best_params['reg_all']
)

model.fit(trainset)

# ---------------------------
# 4 RMSE Evaluation
# ---------------------------

print("\nEvaluating RMSE...")

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)

# ---------------------------
# 5 Precision@K and Recall@K
# ---------------------------

def precision_recall_at_k(predictions, k=5, threshold=4):

    user_est_true = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():

        # sort by predicted rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in top_k
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    return precisions, recalls


print("\nCalculating Precision@K & Recall@K...")

precisions, recalls = precision_recall_at_k(predictions, k=5)

precision_avg = np.mean(list(precisions.values()))
recall_avg = np.mean(list(recalls.values()))

print("Precision@5:", precision_avg)
print("Recall@5:", recall_avg)

# ---------------------------
# 6 Sample Recommendations
# ---------------------------

print("\nSample Recommendations:")

user_id = str(1)

all_movies = ratings_df['movieId'].unique()

rated_movies = ratings_df[ratings_df['userId']==1]['movieId'].values

preds = []

for movie in all_movies:
    if movie not in rated_movies:
        est = model.predict(user_id, movie).est
        preds.append((movie, est))

top_recs = sorted(preds, key=lambda x:x[1], reverse=True)[:5]

print("Top 5 recommendations:", top_recs)