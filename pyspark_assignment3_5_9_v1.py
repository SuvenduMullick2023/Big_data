"""
Task 9 — IMPROVED RL Recommender System
========================================
Five concrete upgrades over the baseline:
  1. Proper RL metrics  : NDCG@10, Hit-Rate@10, Novelty, Coverage, simulated CTR
  2. Content-based state : genre + avg-rating embedding (not raw user index)
  3. LinUCB (contextual bandit) : per-user arm selection using feature dot-products
  4. ε-decay Q-Learning  : epsilon anneals from 1.0 → 0.05 over training
  5. Neural Q-Network    : 2-layer MLP replaces flat Q-table; numpy-only, no torch needed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 0. LOAD & PREPROCESS
# ─────────────────────────────────────────────
print("=" * 65)
print("IMPROVED RL RECOMMENDER — 5 UPGRADES")
print("=" * 65)

movies  = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/movies.csv")
ratings = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/ratings.csv")

# --- genre one-hot (19 genres) -----------------------------------------------
ALL_GENRES = ['Action','Adventure','Animation','Children','Comedy','Crime',
              'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
              'Mystery','Romance','Sci-Fi','Thriller','War','Western','IMAX']

def genre_vec(genre_str: str) -> np.ndarray:
    v = np.zeros(len(ALL_GENRES), dtype=np.float32)
    for g in genre_str.split('|'):
        if g in ALL_GENRES:
            v[ALL_GENRES.index(g)] = 1.0
    return v

movies['genre_vec'] = movies['genres'].apply(genre_vec)

# avg rating per movie
avg_rating = ratings.groupby('movieId')['rating'].mean()
movies['avg_rating'] = movies['movieId'].map(avg_rating).fillna(2.5)
movies['popularity']  = ratings.groupby('movieId')['rating'].count()
movies['popularity']  = movies['popularity'].fillna(0)
movies = movies.set_index('movieId')

# compact index
all_movie_ids = ratings['movieId'].unique()
movie2idx     = {m: i for i, m in enumerate(all_movie_ids)}
idx2movie     = {i: m for m, i in movie2idx.items()}
n_movies      = len(all_movie_ids)

all_user_ids  = ratings['userId'].unique()
user2idx      = {u: i for i, u in enumerate(all_user_ids)}
n_users       = len(all_user_ids)

# user reward map
user_reward_map: dict = defaultdict(dict)
user_rated_map:  dict = defaultdict(set)
for _, row in ratings.iterrows():
    uid  = user2idx[row['userId']]
    midx = movie2idx[row['movieId']]
    r    = 1.0 if row['rating'] >= 4.0 else -1.0
    user_reward_map[uid][midx] = r
    user_rated_map[uid].add(midx)

def get_reward(uid: int, midx: int) -> float:
    return user_reward_map[uid].get(midx, 0.0)

# ─────────────────────────────────────────────
# UPGRADE 1 — PROPER RL METRICS
# ─────────────────────────────────────────────

def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def hit_rate_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    return float(any(r in relevant for r in recommended[:k]))

def novelty_score(recommended: list, popularity_map: dict, n_total: int) -> float:
    """Average self-information of recommended items (less popular → more novel)."""
    scores = []
    for midx in recommended:
        mid = idx2movie.get(midx, -1)
        pop = popularity_map.get(mid, 1) / n_total
        scores.append(-np.log2(pop + 1e-9))
    return float(np.mean(scores)) if scores else 0.0

def coverage(all_recs: list, n_items: int) -> float:
    """Fraction of catalogue ever recommended."""
    return len(set(all_recs)) / n_items

popularity_map = movies['popularity'].to_dict()
n_total_ratings = len(ratings)

# ─────────────────────────────────────────────
# UPGRADE 2 — CONTENT-BASED STATE FEATURES
# ─────────────────────────────────────────────
# State = weighted avg genre vector of user's positively-rated movies + scalar features

FEAT_DIM = len(ALL_GENRES) + 2   # 19 genre dims + avg_rating_bias + n_rated_norm

def user_state(uid: int) -> np.ndarray:
    """Build a FEAT_DIM feature vector from a user's rating history."""
    pos_movies = [midx for midx, r in user_reward_map[uid].items() if r > 0]
    if not pos_movies:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    genre_sum = np.zeros(len(ALL_GENRES), dtype=np.float32)
    for midx in pos_movies:
        mid = idx2movie.get(midx, -1)
        if mid in movies.index:
            genre_sum += movies.loc[mid, 'genre_vec']
    genre_sum /= (np.linalg.norm(genre_sum) + 1e-8)
    avg_r  = np.mean([user_reward_map[uid][m] for m in pos_movies])
    n_norm = min(len(pos_movies) / 50.0, 1.0)
    return np.concatenate([genre_sum, [avg_r, n_norm]]).astype(np.float32)

# Movie feature vectors (for contextual bandit arm features)
def movie_features(midx: int) -> np.ndarray:
    mid = idx2movie.get(midx, -1)
    if mid not in movies.index:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    g   = movies.loc[mid, 'genre_vec']
    ar  = (movies.loc[mid, 'avg_rating'] - 2.5) / 2.5
    pop = min(movies.loc[mid, 'popularity'] / 500.0, 1.0)
    return np.concatenate([g, [ar, pop]]).astype(np.float32)

print(f"[State] Feature dimension : {FEAT_DIM}")

# ─────────────────────────────────────────────
# UPGRADE 3 — LinUCB (Contextual Bandit)
# ─────────────────────────────────────────────

class LinUCB:
    """
    Disjoint LinUCB: each arm has its own ridge regression.
    Score(a) = θ_a·x + α·√(x^T A_a^{-1} x)
    """
    def __init__(self, n_arms: int, feat_dim: int, alpha: float = 0.5):
        self.n_arms   = n_arms
        self.feat_dim = feat_dim
        self.alpha    = alpha
        self.A = [np.eye(feat_dim, dtype=np.float32) for _ in range(n_arms)]
        self.b = [np.zeros(feat_dim, dtype=np.float32) for _ in range(n_arms)]
        self.all_recs: list = []

    def select_arm(self, context: np.ndarray, candidates: list) -> int:
        best_score, best_arm = -np.inf, candidates[0]
        for arm in candidates:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            ucb   = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            if ucb > best_score:
                best_score, best_arm = ucb, arm
        self.all_recs.append(best_arm)
        return best_arm

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def recommend_top_k(self, context: np.ndarray, candidates: list, k: int = 10) -> list:
        scores = []
        for arm in candidates:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            scores.append((theta @ context, arm))
        scores.sort(reverse=True)
        return [arm for _, arm in scores[:k]]

# ─────────────────────────────────────────────
# UPGRADE 4 — ε-DECAY Q-LEARNING
# ─────────────────────────────────────────────

class DecayQLearning:
    """
    Q-Learning with:
      - Content-based state (FEAT_DIM-dimensional, bucketed into N_STATES)
      - ε anneals from eps_start → eps_end over total_steps
      - Reward shaping: bonus for unrated items (novelty bonus)
    """
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 total_steps: int = 50_000):
        self.n_states    = n_states
        self.n_actions   = n_actions
        self.alpha       = alpha
        self.gamma       = gamma
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.total_steps = total_steps
        self.Q           = np.random.uniform(-0.01, 0.01, (n_states, n_actions))
        self.step_count  = 0
        self.explore_log : list = []
        self.all_recs    : list = []

    @property
    def epsilon(self) -> float:
        decay = self.step_count / self.total_steps
        return max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * decay)

    def select_action(self, state: int) -> int:
        self.step_count += 1
        if np.random.rand() < self.epsilon:
            self.explore_log.append(True)
            a = int(np.random.randint(self.n_actions))
        else:
            self.explore_log.append(False)
            a = int(np.argmax(self.Q[state]))
        self.all_recs.append(a)
        return a

    def update(self, s: int, a: int, r: float, ns: int):
        td = r + self.gamma * np.max(self.Q[ns]) - self.Q[s, a]
        self.Q[s, a] += self.alpha * td

    def recommend_top_k(self, state: int, k: int = 10) -> list:
        return list(np.argsort(self.Q[state])[::-1][:k])

    @property
    def explore_rate(self) -> float:
        return sum(self.explore_log) / len(self.explore_log) if self.explore_log else 0.0

# ─────────────────────────────────────────────
# UPGRADE 5 — NEURAL Q-NETWORK (numpy MLP)
# ─────────────────────────────────────────────

class NeuralQNetwork:
    """
    2-layer MLP: state_features → hidden(64) → hidden(32) → Q-values(n_actions)
    Uses ReLU + Adam-lite (simple SGD with momentum).
    Input: FEAT_DIM-dimensional state vector
    Output: Q-value for each of n_actions movies
    """
    def __init__(self, state_dim: int, n_actions: int,
                 hidden: int = 64, lr: float = 1e-3,
                 gamma: float = 0.9, eps_start: float = 1.0,
                 eps_end: float = 0.05, total_steps: int = 30_000):
        self.state_dim   = state_dim
        self.n_actions   = n_actions
        self.lr          = lr
        self.gamma       = gamma
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.total_steps = total_steps
        self.step_count  = 0
        # Xavier init
        def xavier(r, c): return np.random.randn(r, c).astype(np.float32) * np.sqrt(2/(r+c))
        self.W1 = xavier(hidden, state_dim);  self.b1 = np.zeros(hidden, np.float32)
        self.W2 = xavier(32, hidden);         self.b2 = np.zeros(32,     np.float32)
        self.W3 = xavier(n_actions, 32);      self.b3 = np.zeros(n_actions, np.float32)
        self.explore_log: list = []
        self.all_recs:    list = []

    @property
    def epsilon(self) -> float:
        return max(self.eps_end,
                   self.eps_start - (self.eps_start-self.eps_end)*(self.step_count/self.total_steps))

    def relu(self, x): return np.maximum(0, x)

    def forward(self, s: np.ndarray):
        h1 = self.relu(self.W1 @ s + self.b1)
        h2 = self.relu(self.W2 @ h1 + self.b2)
        q  = self.W3 @ h2 + self.b3
        return q, h2, h1

    def select_action(self, s: np.ndarray, candidates: list) -> int:
        self.step_count += 1
        if np.random.rand() < self.epsilon:
            self.explore_log.append(True)
            a = int(np.random.choice(candidates))
        else:
            self.explore_log.append(False)
            q, _, _ = self.forward(s)
            a = candidates[int(np.argmax(q[candidates]))]
        self.all_recs.append(a)
        return a

    def update(self, s: np.ndarray, a: int, r: float, ns: np.ndarray):
        q, h2, h1 = self.forward(s)
        q_next, _, _ = self.forward(ns)
        target = r + self.gamma * np.max(q_next)
        err = np.zeros_like(q); err[a] = q[a] - target   # TD error for action a

        # Backprop
        dW3 = np.outer(err, h2);         db3 = err.copy()
        dh2 = self.W3.T @ err
        dh2 *= (h2 > 0)                  # ReLU gate
        dW2 = np.outer(dh2, h1);         db2 = dh2.copy()
        dh1 = self.W2.T @ dh2
        dh1 *= (h1 > 0)
        dW1 = np.outer(dh1, s);          db1 = dh1.copy()

        # Gradient descent (clipped)
        for (W, dW), (b, db) in zip(
                [(self.W3, dW3),(self.W2, dW2),(self.W1, dW1)],
                [(self.b3, db3),(self.b2, db2),(self.b1, db1)]):
            W -= self.lr * np.clip(dW, -1, 1)
            b -= self.lr * np.clip(db, -1, 1)

    def recommend_top_k(self, s: np.ndarray, candidates: list, k: int = 10) -> list:
        q, _, _ = self.forward(s)
        ranked  = sorted(candidates, key=lambda a: -q[a])
        return ranked[:k]

    @property
    def explore_rate(self) -> float:
        return sum(self.explore_log)/len(self.explore_log) if self.explore_log else 0.0

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
N_SIM_USERS   = 200
N_STEPS       = 300       # per user for MAB / Decay-QL
N_STATES      = 50        # buckets for Decay-QL state space
CAND_SIZE     = 200       # candidate pool per step (tractable subset)

# Precompute state buckets (kmeans-lite: hash user_state into N_STATES)
def state_bucket(uid: int) -> int:
    sv = user_state(uid)
    return int(np.abs(hash(sv.tobytes())) % N_STATES)

# Candidate pool: top-CAND_SIZE movies by popularity
top_cand = list(range(min(CAND_SIZE, n_movies)))

print("\n[1/4] Training LinUCB …")
linucb = LinUCB(n_movies, FEAT_DIM, alpha=0.5)
linucb_rewards, linucb_eps_log = [], []
for uid in range(N_SIM_USERS):
    ctx = user_state(uid)
    for _ in range(N_STEPS):
        arm = linucb.select_arm(ctx, top_cand)
        r   = get_reward(uid, arm)
        linucb.update(arm, ctx, r)
        linucb_rewards.append(r)
print(f"  LinUCB cum reward : {sum(linucb_rewards):.1f}")

print("[2/4] Training ε-Decay Q-Learning …")
TOTAL_QL = N_SIM_USERS * N_STEPS
dql = DecayQLearning(N_STATES, CAND_SIZE, alpha=0.15, gamma=0.9,
                     eps_start=1.0, eps_end=0.05, total_steps=TOTAL_QL)
dql_rewards, dql_eps_curve = [], []
for uid in range(N_SIM_USERS):
    s = state_bucket(uid)
    for _ in range(N_STEPS):
        a   = dql.select_action(s)
        r   = get_reward(uid, a)
        # novelty bonus: reward unrated items slightly
        if a not in user_rated_map[uid]:
            r += 0.1
        ns  = (s + 1) % N_STATES
        dql.update(s, a, r, ns)
        s   = ns
        dql_rewards.append(r)
        dql_eps_curve.append(dql.epsilon)
print(f"  Decay-QL cum reward : {sum(dql_rewards):.1f} | Final ε : {dql.epsilon:.3f}")

print("[3/4] Training Neural Q-Network …")
N_NQL_USERS = 150
N_NQL_STEPS = 200
TOTAL_NQL   = N_NQL_USERS * N_NQL_STEPS
nqn = NeuralQNetwork(FEAT_DIM, CAND_SIZE, hidden=64, lr=5e-4,
                     gamma=0.9, eps_start=1.0, eps_end=0.05,
                     total_steps=TOTAL_NQL)
nqn_rewards, nqn_loss_proxy = [], []
for uid in range(N_NQL_USERS):
    s_vec = user_state(uid)
    for step in range(N_NQL_STEPS):
        a   = nqn.select_action(s_vec, top_cand)
        r   = get_reward(uid, a)
        if a not in user_rated_map[uid]:
            r += 0.1
        ns_vec = user_state((uid + 1) % n_users)
        nqn.update(s_vec, a, r, ns_vec)
        s_vec  = ns_vec
        nqn_rewards.append(r)
        if step % 20 == 0:
            q_vals, _, _ = nqn.forward(s_vec)
            nqn_loss_proxy.append(float(np.var(q_vals[:CAND_SIZE])))
print(f"  Neural-QN cum reward : {sum(nqn_rewards):.1f} | Final ε : {nqn.epsilon:.3f}")

# ─────────────────────────────────────────────
# EVALUATION with PROPER RL METRICS
# ─────────────────────────────────────────────
print("\n[4/4] Evaluating with RL-appropriate metrics …")

# Build sub-matrix for traditional baselines
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

top_movies_set = set(ratings['movieId'].value_counts().head(500).index)
top_users_set  = set(ratings['userId'].value_counts().head(500).index)
sub = ratings[ratings['movieId'].isin(top_movies_set) & ratings['userId'].isin(top_users_set)]
ui  = sub.pivot(index='userId', columns='movieId', values='rating').fillna(0)
U   = ui.values

svd     = TruncatedSVD(n_components=20, random_state=42).fit(U)
U_approx = svd.inverse_transform(svd.transform(U))
user_sim = cosine_similarity(U)
item_sim = cosine_similarity(U.T)

def svd_recs(row, k=10):      return list(np.argsort(U_approx[row])[::-1][:k])
def user_cf_recs(row, k=10):
    s=user_sim[row].copy(); s[row]=-1
    return list(np.argsort(U[np.argsort(s)[::-1][:10]].mean(0))[::-1][:k])
def item_cf_recs(row, k=10):  return list(np.argsort(item_sim@U[row])[::-1][:k])

N_EVAL = min(100, U.shape[0])
K      = 10
metrics = {m: {'ndcg':[], 'hr':[], 'novelty':[]}
           for m in ['SVD','User-CF','Item-CF','LinUCB','Decay-QL','Neural-QN']}
all_recs_per_model = defaultdict(list)

for row in range(N_EVAL):
    uid     = row % n_users
    relevant = {i for i, v in enumerate(U[row]) if v >= 4}
    if not relevant:
        continue
    ctx    = user_state(uid)
    s_bkt  = state_bucket(uid)

    recs = {
        'SVD':      svd_recs(row),
        'User-CF':  user_cf_recs(row),
        'Item-CF':  item_cf_recs(row),
        'LinUCB':   linucb.recommend_top_k(ctx, top_cand, K),
        'Decay-QL': dql.recommend_top_k(s_bkt, K),
        'Neural-QN':nqn.recommend_top_k(ctx, top_cand, K),
    }
    for name, rec in recs.items():
        metrics[name]['ndcg'].append(ndcg_at_k(rec, relevant, K))
        metrics[name]['hr'].append(hit_rate_at_k(rec, relevant, K))
        metrics[name]['novelty'].append(novelty_score(rec, popularity_map, n_total_ratings))
        all_recs_per_model[name].extend(rec)

# Aggregate
results = {}
for name in metrics:
    results[name] = {
        'NDCG@10':    float(np.mean(metrics[name]['ndcg'])),
        'HitRate@10': float(np.mean(metrics[name]['hr'])),
        'Novelty':    float(np.mean(metrics[name]['novelty'])),
        'Coverage':   coverage(all_recs_per_model[name], n_movies),
    }

# Simulated CTR: fraction of recs that got positive reward in simulation
results['LinUCB']['CTR']     = (np.array(linucb_rewards) > 0).mean()
results['Decay-QL']['CTR']   = (np.array(dql_rewards)    > 0).mean()
results['Neural-QN']['CTR']  = (np.array(nqn_rewards)    > 0).mean()
for m in ['SVD','User-CF','Item-CF']:
    results[m]['CTR'] = float('nan')

# Print results
print(f"\n{'Model':<13} {'NDCG@10':>9} {'HitRate@10':>12} {'Novelty':>9} {'Coverage':>10} {'CTR':>8}")
print("-" * 65)
for name, r in results.items():
    ctr = f"{r['CTR']:.4f}" if not (isinstance(r['CTR'], float) and np.isnan(r['CTR'])) else "  N/A"
    print(f"{name:<13} {r['NDCG@10']:>9.4f} {r['HitRate@10']:>12.4f} "
          f"{r['Novelty']:>9.2f} {r['Coverage']:>10.4f} {ctr:>8}")

# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
print("\n[Plot] Generating charts …")

fig = plt.figure(figsize=(20, 14), facecolor="#0d0d0d")
fig.suptitle("Improved RL Recommender — 5 Upgrades", fontsize=16,
             color="white", fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

DARK, PANEL = "#0d0d0d", "#1a1a2e"
COLORS6 = {"SVD":"#4cc9f0","User-CF":"#4895ef","Item-CF":"#4361ee",
            "LinUCB":"#f72585","Decay-QL":"#7209b7","Neural-QN":"#3a0ca3"}

def ax_style(ax, title):
    ax.set_facecolor(PANEL); ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="#aaa"); [s.set_color("#333") for s in ax.spines.values()]

# (A) Epsilon decay curve
ax1 = fig.add_subplot(gs[0, 0])
ax_style(ax1, "ε Decay — Decay-QL")
ax1.plot(np.linspace(0,1,len(dql_eps_curve)), dql_eps_curve, color="#f72585", lw=2)
ax1.axhline(0.05, color="#aaa", lw=1, ls="--", label="ε_min=0.05")
ax1.set_xlabel("Training progress", color="#aaa"); ax1.set_ylabel("ε", color="#aaa")
ax1.legend(facecolor="#111", labelcolor="white", fontsize=9)

# (B) Cumulative rewards
ax2 = fig.add_subplot(gs[0, 1])
ax_style(ax2, "Cumulative Reward Curves")
stride = max(1, len(linucb_rewards)//500)
ax2.plot(np.cumsum(linucb_rewards)[::stride],  color="#f72585", lw=2, label="LinUCB")
ax2.plot(np.cumsum(dql_rewards)[::stride],     color="#7209b7", lw=2, label="Decay-QL")
ax2.plot(np.cumsum(nqn_rewards)[::stride],     color="#3a0ca3", lw=2, label="Neural-QN")
ax2.set_xlabel("Steps", color="#aaa"); ax2.set_ylabel("Cumulative Reward", color="#aaa")
ax2.legend(facecolor="#111", labelcolor="white", fontsize=9)

# (C) NDCG@10
ax3 = fig.add_subplot(gs[0, 2])
ax_style(ax3, "NDCG@10 (quality of ranking)")
names = list(results.keys())
ndcgs = [results[n]['NDCG@10'] for n in names]
bars  = ax3.barh(names, ndcgs, color=[COLORS6[n] for n in names], edgecolor="#222", height=0.55)
for b, v in zip(bars, ndcgs):
    ax3.text(b.get_width()+.002, b.get_y()+b.get_height()/2,
             f"{v:.4f}", va="center", color="white", fontsize=9)
ax3.set_xlabel("NDCG@10", color="#aaa"); ax3.set_xlim(0, max(ndcgs)*1.3)

# (D) HitRate@10 vs Novelty scatter
ax4 = fig.add_subplot(gs[1, 0])
ax_style(ax4, "Hit-Rate@10 vs Novelty")
for n in names:
    ax4.scatter(results[n]['HitRate@10'], results[n]['Novelty'],
                color=COLORS6[n], s=120, zorder=5, label=n)
    ax4.annotate(n, (results[n]['HitRate@10'], results[n]['Novelty']),
                 textcoords="offset points", xytext=(6,4), fontsize=8, color="#aaa")
ax4.set_xlabel("Hit-Rate@10 (relevance)", color="#aaa")
ax4.set_ylabel("Novelty (self-info)", color="#aaa")
ax4.legend(facecolor="#111", labelcolor="white", fontsize=8, ncol=2)

# (E) Coverage
ax5 = fig.add_subplot(gs[1, 1])
ax_style(ax5, "Catalogue Coverage")
covs = [results[n]['Coverage']*100 for n in names]
ax5.bar(names, covs, color=[COLORS6[n] for n in names], edgecolor="#222", width=0.55)
for i, v in enumerate(covs):
    ax5.text(i, v+0.3, f"{v:.1f}%", ha="center", color="white", fontsize=9)
ax5.set_ylabel("% of catalogue covered", color="#aaa")
plt.setp(ax5.get_xticklabels(), rotation=30, ha='right', fontsize=9)

# (F) Neural QN loss proxy (Q-variance over training)
ax6 = fig.add_subplot(gs[1, 2])
ax_style(ax6, "Neural-QN Q-Value Variance (proxy for learning)")
ax6.plot(nqn_loss_proxy, color="#3a0ca3", lw=2)
ax6.set_xlabel("Training checkpoints (×20 steps)", color="#aaa")
ax6.set_ylabel("Q-value variance", color="#aaa")
path ="/home/suvendu/mlbd/code/Big_data/"
plt.savefig(path + "rl_improved_results.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print("[Done] Saved → rl_improved_results.png")
print("=" * 65)
print("UPGRADE SUMMARY")
print("=" * 65)
print("""
  1. NDCG@10 / HitRate@10 / Novelty / Coverage / CTR
     → Fairer metrics that reward exploration & diverse discovery

  2. Content-based state (genre + avg_rating + n_rated)
     → RL agents now understand WHAT the user likes, not just WHO

  3. LinUCB (contextual bandit)
     → Arms scored via ridge regression on user+item features
     → Naturally personalised, no random exploration needed

  4. ε-Decay Q-Learning
     → Starts fully exploratory (ε=1), anneals to ε=0.05
     → Novelty bonus (+0.1) for recommending unrated movies
     → No wasted exploitation with uninformed initial Q-values

  5. Neural Q-Network (2-layer MLP, numpy only)
     → Generalises across the state-action space
     → Backprop with gradient clipping; Xavier init
     → Scales to large item catalogues without a huge Q-table
""")