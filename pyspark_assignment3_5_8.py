import pandas as pd
import numpy as np
# Part 5: Learning-Based Recommender Systems
# Task 8: Content-Based Filtering with a Neural Network


# -------------------------
# 1. Load data
# -------------------------
movies = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/movies.csv")
ratings = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/ratings.csv")

# -------------------------
# 2. Movie features
# -------------------------
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
movies['year'] = movies['year'].fillna(movies['year'].median())

# Genres → multi-hot
genres = movies['genres'].str.get_dummies(sep='|')
movies = pd.concat([movies, genres], axis=1)

# Avg movie rating
movie_avg_rating = ratings.groupby('movieId')['rating'].mean()
movies = movies.merge(movie_avg_rating, on='movieId', how='left')
movies.rename(columns={'rating': 'avg_rating'}, inplace=True)

movies.fillna(0, inplace=True)

# -------------------------
# 3. User features
# -------------------------
data = ratings.merge(movies[['movieId'] + list(genres.columns)], on='movieId')

user_genre_pref = data.groupby('userId')[genres.columns].apply(
    lambda x: np.average(x, axis=0, weights=data.loc[x.index, 'rating'])
)

user_features = pd.DataFrame(
    user_genre_pref.tolist(),
    index=user_genre_pref.index,
    columns=genres.columns
)

user_features.fillna(0, inplace=True)

# 🔥 IMPORTANT FIXES
user_features = user_features.reset_index()   # make userId a column
user_features = user_features.add_prefix("user_")
user_features.rename(columns={"user_userId": "userId"}, inplace=True)

# -------------------------
# 4. Merge everything
# -------------------------
dataset = ratings.merge(movies, on='movieId')
dataset = dataset.merge(user_features, on='userId')

# -------------------------
# 5. Define columns
# -------------------------
movie_cols = list(genres.columns) + ['year', 'avg_rating']
user_cols = ["user_" + col for col in genres.columns]

# -------------------------
# 6. Extract final matrices
# -------------------------
X_user = dataset[user_cols].values
X_movie = dataset[movie_cols].values
y = dataset['rating'].values

# -------------------------
# 7. Sanity check
# -------------------------
print("User shape:", X_user.shape)
print("Movie shape:", X_movie.shape)
print("Target shape:", y.shape)


from sklearn.preprocessing import StandardScaler

user_scaler = StandardScaler()
movie_scaler = StandardScaler()

X_user = user_scaler.fit_transform(X_user)
X_movie = movie_scaler.fit_transform(X_movie)

X_user = X_user.astype('float32')
X_movie = X_movie.astype('float32')
y = y.astype('float32')

print("User shape:", X_user.shape)
print("Movie shape:", X_movie.shape)
print("Ratings shape:", y.shape)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

def precision_recall_at_k(results, k=10, threshold=4.0):
    """
    results: list of (userId, movieId, actual_rating, predicted_rating)
    """

    user_true = defaultdict(set)
    user_pred = defaultdict(list)

    # Build user-wise data
    for user, movie, actual, pred in results:
        if actual >= threshold:
            user_true[user].add(movie)

        user_pred[user].append((movie, pred))

    precisions = []
    recalls = []

    for user in user_pred:
        # Sort by predicted rating
        ranked = sorted(user_pred[user], key=lambda x: x[1], reverse=True)

        top_k = [m for m, _ in ranked[:k]]

        true_items = user_true[user]

        if len(true_items) == 0:
            continue

        hits = len(set(top_k) & true_items)

        precision = hits / k
        recall = hits / len(true_items)

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)




class MovieLensDataset(Dataset):
    def __init__(self, user_features, movie_features, ratings, user_ids, movie_ids):
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.movie_features = torch.tensor(movie_features, dtype=torch.float32)
        self.ratings = torch.tensor(ratings, dtype=torch.float32).view(-1, 1)
        self.user_ids = user_ids
        self.movie_ids = movie_ids

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_features[idx],
            self.movie_features[idx],
            self.ratings[idx],
            self.user_ids[idx],
            self.movie_ids[idx]
        )
        

class RecommenderNet(nn.Module):
    def __init__(self, num_user_features, num_movie_features, embedding_dim=16):
        super(RecommenderNet, self).__init__()

        # USER NETWORK
        self.user_net = nn.Sequential(
            nn.Linear(num_user_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

        # MOVIE NETWORK
        self.movie_net = nn.Sequential(
            nn.Linear(num_movie_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

        # FINAL PREDICTION LAYER
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_input, movie_input):
        user_emb = self.user_net(user_input)
        movie_emb = self.movie_net(movie_input)

        combined = torch.cat([user_emb, movie_emb], dim=1)
        output = self.fc(combined)

        return output        
    

user_ids = dataset['userId'].values
movie_ids = dataset['movieId'].values

from sklearn.model_selection import train_test_split

X_user_train, X_user_val, X_movie_train, X_movie_val, y_train, y_val, \
user_train, user_val, movie_train, movie_val = train_test_split(
    X_user, X_movie, y, user_ids, movie_ids,
    test_size=0.2, random_state=42
)



train_dataset = MovieLensDataset(
    X_user_train, X_movie_train, y_train,
    user_train, movie_train
)

val_dataset = MovieLensDataset(
    X_user_val, X_movie_val, y_val,
    user_val, movie_val
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, pin_memory=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RecommenderNet(
    num_user_features=X_user.shape[1],
    num_movie_features=X_movie.shape[1]
).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for user, movie, rating, _, _ in train_loader:   
            user = user.to(device, non_blocking=True)
            movie = movie.to(device, non_blocking=True)
            rating = rating.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(user, movie)

            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for user, movie, rating, _, _ in val_loader:  
                user = user.to(device, non_blocking=True)
                movie = movie.to(device, non_blocking=True)
                rating = rating.to(device, non_blocking=True)

                output = model(user, movie)
                loss = criterion(output, rating)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: Train={train_loss/len(train_loader):.4f}, Val={val_loss/len(val_loader):.4f}")
        
        

        


def evaluate(model, val_loader):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for user, movie, rating, _, _ in val_loader:  
            user = user.to(device)
            movie = movie.to(device)

            output = model(user, movie)

            preds.extend(output.cpu().numpy())
            actuals.extend(rating.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    rmse = np.sqrt(((preds - actuals) ** 2).mean())
    return rmse



def get_predictions(model, loader):
    model.eval()
    results = []

    with torch.no_grad():
        for user, movie, rating, user_id, movie_id in loader:
            user = user.to(device)
            movie = movie.to(device)

            preds = model(user, movie).cpu().numpy().flatten()

            for u, m, r, p in zip(user_id, movie_id, rating.numpy(), preds):
                results.append((u, m, r, p))

    return results

train_model(model, train_loader, val_loader, epochs=20)

rmse = evaluate(model, val_loader)
print("RMSE for val dataset:", rmse)

# Ranking evaluation
results_val = get_predictions(model, val_loader)

p_at_5, r_at_5 = precision_recall_at_k(results_val, k=5)
p_at_10, r_at_10 = precision_recall_at_k(results_val, k=10)

print("\nValidation Set Evaluation")
print(f"Precision@5: {p_at_5:.4f}, Recall@5: {r_at_5:.4f}")
print(f"Precision@10: {p_at_10:.4f}, Recall@10: {r_at_10:.4f}")


# Task 12: Model-Agnostic Explainability (For Deep Learning Models)
# Use tools like LIME (Local Interpretable Model-agnostic Explanations) to break down neural network decisions.


# -------------------------
# 🔥 LIME EXPLAINABILITY
# -------------------------
from lime.lime_tabular import LimeTabularExplainer

# Combine features (IMPORTANT)
X_combined = np.concatenate([X_user, X_movie], axis=1)

# Feature names
feature_names = (
    ["user_" + col for col in genres.columns] +   # user features
    list(genres.columns) +                        # movie genres
    ['year', 'avg_rating']                        # movie metadata
)

# -------------------------
# Predict function for LIME
# -------------------------
def predict_fn(x):
    model.eval()

    x = torch.tensor(x, dtype=torch.float32).to(device)

    # Split back into user + movie
    user_part = x[:, :X_user.shape[1]]
    movie_part = x[:, X_user.shape[1]:]

    with torch.no_grad():
        preds = model(user_part, movie_part).cpu().numpy()

    return preds

# -------------------------
# Create Explainer
# -------------------------
explainer = LimeTabularExplainer(
    X_combined,
    feature_names=feature_names,
    mode='regression'
)

# -------------------------
# Explain ONE prediction
# -------------------------
sample_idx = 0  # change this to explore different cases

exp = explainer.explain_instance(
    X_combined[sample_idx],
    predict_fn,
    num_features=10
)

# -------------------------
# Print explanation
# -------------------------
print("\n🔍 LIME Explanation for sample:", sample_idx)
for feature, weight in exp.as_list():
    print(f"{feature}: {weight:.4f}")
    
    
    
# pick a validation sample
user, movie, rating, uid, mid = next(iter(val_loader))

idx = 0
combined_sample = np.concatenate([
    user[idx].numpy(),
    movie[idx].numpy()
])

exp = explainer.explain_instance(
    combined_sample,
    predict_fn,
    num_features=10
)

print(f"\nUser {uid[idx]} - Movie {mid[idx]}")
print("Actual rating:", rating[idx].item())

for f, w in exp.as_list():
    print(f"{f}: {w:.4f}")    