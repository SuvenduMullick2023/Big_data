# Traditional Content-Based Filtering (Cosine Similarity)
# Evaluate performance and compare with traditional content-based filtering (cosine similarity on genres). Does the neural
# model capture more complex user preferences than standard TF-IDF-based filtering?

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# -------------------------
# 1. Load Data
# -------------------------
movies = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/movies.csv")
ratings = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/ratings.csv")

# -------------------------
# 2. Train/Test Split (80/20)
# -------------------------
np.random.seed(42)
ratings['is_test'] = np.random.rand(len(ratings)) < 0.2

train_ratings = ratings[ratings['is_test'] == False]
test_ratings = ratings[ratings['is_test'] == True]

print(f"Train: {len(train_ratings)}, Test: {len(test_ratings)}")

# -------------------------
# 3. Movie Features (Genres)
# -------------------------
genres = movies['genres'].str.get_dummies(sep='|')
movie_genre_matrix = genres.values.astype(np.float32)

# Mappings
movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies['movieId'])}
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}
n_movies = len(movies)

# -------------------------
# 4. Compute Similarity Matrix
# -------------------------
print("Computing similarity matrix...")
similarity_matrix = cosine_similarity(movie_genre_matrix)

# -------------------------
# 5. CBF Predictions with Proper Evaluation
# -------------------------
def get_cbf_predictions_with_split(train_ratings, test_ratings, similarity_matrix, 
                                    movie_id_to_idx, idx_to_movie_id, k=10, threshold=4.0):
    results = []
    
    # Get test set ground truth per user
    test_user_movies = test_ratings[test_ratings['rating'] >= threshold].groupby('userId')['movieId'].apply(set).to_dict()
    
    # Train: get liked movies per user
    train_user_liked = train_ratings[train_ratings['rating'] >= threshold].groupby('userId')['movieId'].apply(list).to_dict()
    
    # All movies user has seen (train + test) to exclude from recommendations
    all_user_movies = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()
    
    # Process each user who has test data
    for user in test_user_movies.keys():
        if user not in train_user_liked or len(train_user_liked[user]) == 0:
            continue
        
        liked_indices = [movie_id_to_idx[m] for m in train_user_liked[user] if m in movie_id_to_idx]
        if len(liked_indices) == 0:
            continue
        
        # Aggregate similarity scores from liked movies
        sim_scores = similarity_matrix[liked_indices]
        aggregated_scores = np.sum(sim_scores, axis=0)
        
        # Exclude ALL movies user has already seen (train + test)
        if user in all_user_movies:
            rated_indices = [movie_id_to_idx[m] for m in all_user_movies[user] if m in movie_id_to_idx]
            aggregated_scores[rated_indices] = -np.inf
        
        # Get top-k recommendations
        top_k_indices = np.argsort(aggregated_scores)[::-1][:k]
        top_k_movie_ids = [idx_to_movie_id[idx] for idx in top_k_indices if idx in idx_to_movie_id]
        
        # Ground truth: test set movies user liked
        test_set = test_user_movies.get(user, set())
        
        for m_id in top_k_movie_ids:
            actual = 1 if m_id in test_set else 0
            results.append((user, m_id, actual, aggregated_scores[movie_id_to_idx[m_id]]))
    
    return results

# -------------------------
# 6. Precision/Recall Function
# -------------------------
def precision_recall_at_k(results, k=5):
    user_results = defaultdict(list)
    
    for user, movie, actual, score in results:
        user_results[user].append((movie, actual, score))
    
    precisions = []
    recalls = []
    
    for user, recs in user_results.items():
        recs_sorted = sorted(recs, key=lambda x: x[2], reverse=True)[:k]
        
        hits = sum(1 for _, actual, _ in recs_sorted if actual == 1)
        
        precision = hits / len(recs_sorted) if len(recs_sorted) > 0 else 0
        recall = hits / k if k > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.mean(precisions), np.mean(recalls)

# -------------------------
# 7. Run Evaluation
# -------------------------
print("Generating CBF predictions...")
results_cbf = get_cbf_predictions_with_split(train_ratings, test_ratings, similarity_matrix, 
                                              movie_id_to_idx, idx_to_movie_id, k=10)

print("Calculating metrics...")
p5_cbf, r5_cbf = precision_recall_at_k(results_cbf, k=5)
p10_cbf, r10_cbf = precision_recall_at_k(results_cbf, k=10)

print("\n" + "="*50)
print("Cosine Similarity Model (Content-Based)")
print("="*50)
print(f"Precision@5:  {p5_cbf:.4f}, Recall@5:  {r5_cbf:.4f}")
print(f"Precision@10: {p10_cbf:.4f}, Recall@10: {r10_cbf:.4f}")
print("="*50)