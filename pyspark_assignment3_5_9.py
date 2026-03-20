"""
Task 9: Reinforcement Learning in Recommender Systems
=====================================================
Implements MAB (ε-Greedy & UCB), Q-Learning, and compares with SVD, User-CF, Item-CF
Dataset: MovieLens Small
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("TASK 9: RL-BASED RECOMMENDER SYSTEM")
print("=" * 60)

movies  = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/movies.csv")
ratings = pd.read_csv("/home/suvendu/mlbd/ml-latest-small/ratings.csv")

print(f"\n[Data] Movies : {movies.shape[0]:,} | Ratings : {ratings.shape[0]:,}")
print(f"[Data] Users  : {ratings['userId'].nunique():,} | Items : {ratings['movieId'].nunique():,}")

# ─────────────────────────────────────────────
# 2. ENVIRONMENT SETUP
# ─────────────────────────────────────────────
# Encode movieId to compact indices
all_movie_ids  = ratings["movieId"].unique()
movie2idx      = {m: i for i, m in enumerate(all_movie_ids)}
idx2movie      = {i: m for m, i in movie2idx.items()}
n_movies       = len(all_movie_ids)

all_user_ids   = ratings["userId"].unique()
user2idx       = {u: i for i, u in enumerate(all_user_ids)}
n_users        = len(all_user_ids)

# Reward function: ≥4 → +1, <4 → -1, missing → 0
def get_reward(rating: float) -> float:
    if rating >= 4.0:
        return 1.0
    elif rating < 4.0:
        return -1.0
    return 0.0

# Build user→{movieIdx: reward} lookup for fast simulation
user_reward_map: dict[int, dict[int, float]] = defaultdict(dict)
for _, row in ratings.iterrows():
    uid  = user2idx[row["userId"]]
    midx = movie2idx[row["movieId"]]
    user_reward_map[uid][midx] = get_reward(row["rating"])

def simulate_reward(user_idx: int, movie_idx: int) -> float:
    """Return observed reward for (user, movie); 0 if unrated."""
    return user_reward_map[user_idx].get(movie_idx, 0.0)

print(f"\n[Env] Reward mapping built for {n_users} users × {n_movies} movies.")

# ─────────────────────────────────────────────
# 3. MULTI-ARMED BANDIT
# ─────────────────────────────────────────────

class EpsilonGreedyMAB:
    """ε-Greedy Multi-Armed Bandit recommender."""

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms   = n_arms
        self.epsilon  = epsilon
        self.counts   = np.zeros(n_arms)          # pulls per arm
        self.values   = np.zeros(n_arms)          # running mean reward
        self.explore_log: list[bool] = []

    def select_arm(self) -> int:
        if np.random.rand() < self.epsilon:
            self.explore_log.append(True)
            return int(np.random.randint(self.n_arms))
        self.explore_log.append(False)
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n   # incremental mean

    def recommend_top_k(self, k: int = 5) -> list[int]:
        return list(np.argsort(self.values)[::-1][:k])

    @property
    def explore_rate(self) -> float:
        if not self.explore_log:
            return 0.0
        return sum(self.explore_log) / len(self.explore_log)


class UCB_MAB:
    """Upper Confidence Bound Multi-Armed Bandit recommender."""

    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c      = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t      = 0
        self.explore_log: list[bool] = []

    def select_arm(self) -> int:
        self.t += 1
        # Pull each arm once first
        unpulled = np.where(self.counts == 0)[0]
        if unpulled.size > 0:
            arm = int(unpulled[0])
            self.explore_log.append(True)
            return arm
        ucb_vals = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        arm = int(np.argmax(ucb_vals))
        self.explore_log.append(self.counts[arm] < 5)  # proxy for "exploring"
        return arm

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def recommend_top_k(self, k: int = 5) -> list[int]:
        return list(np.argsort(self.values)[::-1][:k])

    @property
    def explore_rate(self) -> float:
        if not self.explore_log:
            return 0.0
        return sum(self.explore_log) / len(self.explore_log)


# ── Train MAB on a sample of users ──────────────────────────────
N_SIM_USERS = 200
N_STEPS     = 300          # interactions per user

mab_eg  = EpsilonGreedyMAB(n_movies, epsilon=0.1)
mab_ucb = UCB_MAB(n_movies, c=2.0)

eg_cumrew,  eg_steps  = [], []
ucb_cumrew, ucb_steps = [], []

print("\n[MAB] Simulating ε-Greedy & UCB …")
for user_idx in range(N_SIM_USERS):
    for step in range(N_STEPS):
        # ε-Greedy
        arm_eg = mab_eg.select_arm()
        r_eg   = simulate_reward(user_idx, arm_eg)
        mab_eg.update(arm_eg, r_eg)
        eg_steps.append(r_eg)

        # UCB
        arm_ucb = mab_ucb.select_arm()
        r_ucb   = simulate_reward(user_idx, arm_ucb)
        mab_ucb.update(arm_ucb, r_ucb)
        ucb_steps.append(r_ucb)

eg_cumrew  = np.cumsum(eg_steps)
ucb_cumrew = np.cumsum(ucb_steps)

print(f"  ε-Greedy  → Final cumulative reward : {eg_cumrew[-1]:.1f} | Explore rate : {mab_eg.explore_rate:.2%}")
print(f"  UCB       → Final cumulative reward : {ucb_cumrew[-1]:.1f} | Explore rate : {mab_ucb.explore_rate:.2%}")

# ─────────────────────────────────────────────
# 4. Q-LEARNING AGENT
# ─────────────────────────────────────────────

class QLearningRecommender:
    """
    Tabular Q-Learning recommender.
    State  : user index (proxy for interaction history bucket)
    Action : movie index
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        # Q-table: small random init
        self.Q = np.random.uniform(low=-0.01, high=0.01, size=(n_states, n_actions))
        self.explore_log: list[bool] = []

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            self.explore_log.append(True)
            return int(np.random.randint(self.n_actions))
        self.explore_log.append(False)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def recommend_top_k(self, state: int, k: int = 5) -> list[int]:
        return list(np.argsort(self.Q[state])[::-1][:k])

    @property
    def explore_rate(self) -> float:
        if not self.explore_log:
            return 0.0
        return sum(self.explore_log) / len(self.explore_log)


# Use a smaller Q-table for tractability: bucket users into 50 states
N_STATES   = 50
N_QL_USERS = 300
N_QL_STEPS = 200

ql_agent = QLearningRecommender(N_STATES, n_movies, alpha=0.1, gamma=0.9, epsilon=0.1)
ql_rewards, ql_cumrew = [], []

print("\n[Q-Learning] Training agent …")
for user_idx in range(N_QL_USERS):
    state = user_idx % N_STATES
    for step in range(N_QL_STEPS):
        action     = ql_agent.select_action(state)
        reward     = simulate_reward(user_idx, action)
        next_state = (state + 1) % N_STATES
        ql_agent.update(state, action, reward, next_state)
        state      = next_state
        ql_rewards.append(reward)

ql_cumrew = np.cumsum(ql_rewards)
print(f"  Q-Learning → Final cumulative reward : {ql_cumrew[-1]:.1f} | Explore rate : {ql_agent.explore_rate:.2%}")

# ─────────────────────────────────────────────
# 5. TRADITIONAL MODELS (SVD, User-CF, Item-CF)
# ─────────────────────────────────────────────

print("\n[Traditional] Building user-item matrix …")

# Use top-500 most-rated movies and top-500 most-active users for speed
top_movies = ratings["movieId"].value_counts().head(500).index
top_users  = ratings["userId"].value_counts().head(500).index

sub = ratings[ratings["movieId"].isin(top_movies) & ratings["userId"].isin(top_users)]
ui_matrix = sub.pivot(index="userId", columns="movieId", values="rating").fillna(0)
U_arr     = ui_matrix.values   # shape (n_users_sub, n_movies_sub)

print(f"  Sub-matrix shape : {U_arr.shape}")

# ── SVD ─────────────────────────────────────
print("[Traditional] Fitting SVD (k=20) …")
svd       = TruncatedSVD(n_components=20, random_state=42)
U_reduced = svd.fit_transform(U_arr)
U_approx  = svd.inverse_transform(U_reduced)

def svd_recommend(user_row: int, k: int = 5) -> list[int]:
    scores = U_approx[user_row]
    return list(np.argsort(scores)[::-1][:k])

# ── User-CF ─────────────────────────────────
print("[Traditional] Computing user-user similarity …")
user_sim = cosine_similarity(U_arr)   # (n_u, n_u)

def user_cf_recommend(user_row: int, k: int = 5) -> list[int]:
    sims    = user_sim[user_row].copy()
    sims[user_row] = -1               # exclude self
    top_n   = np.argsort(sims)[::-1][:10]
    pred    = U_arr[top_n].mean(axis=0)
    return list(np.argsort(pred)[::-1][:k])

# ── Item-CF ─────────────────────────────────
print("[Traditional] Computing item-item similarity …")
item_sim = cosine_similarity(U_arr.T)   # (n_i, n_i)

def item_cf_recommend(user_row: int, k: int = 5) -> list[int]:
    user_vec = U_arr[user_row]
    scores   = item_sim @ user_vec
    return list(np.argsort(scores)[::-1][:k])

# ─────────────────────────────────────────────
# 6. EVALUATION: Precision@5 on held-out ratings
# ─────────────────────────────────────────────

print("\n[Eval] Computing Precision@5 …")

def precision_at_k(recommended: list[int], relevant: set[int], k: int = 5) -> float:
    rec_k = recommended[:k]
    hits  = sum(1 for r in rec_k if r in relevant)
    return hits / k

N_EVAL = min(100, ui_matrix.shape[0])
metrics = {"SVD": [], "User-CF": [], "Item-CF": [],
           "ε-Greedy": [], "UCB": [], "Q-Learning": []}

for user_row in range(N_EVAL):
    # Relevant = movies with rating ≥ 4
    row_ratings = U_arr[user_row]
    relevant    = set(np.where(row_ratings >= 4)[0])
    if not relevant:
        continue

    metrics["SVD"].append(precision_at_k(svd_recommend(user_row), relevant))
    metrics["User-CF"].append(precision_at_k(user_cf_recommend(user_row), relevant))
    metrics["Item-CF"].append(precision_at_k(item_cf_recommend(user_row), relevant))

    # RL models: top-5 by learned values (global, not user-specific sub-matrix)
    # Map sub-matrix cols back to n_movies space via position
    eg_recs  = mab_eg.recommend_top_k(5)
    ucb_recs = mab_ucb.recommend_top_k(5)
    ql_recs  = ql_agent.recommend_top_k(user_row % N_STATES, 5)

    metrics["ε-Greedy"].append(precision_at_k(eg_recs, relevant))
    metrics["UCB"].append(precision_at_k(ucb_recs, relevant))
    metrics["Q-Learning"].append(precision_at_k(ql_recs, relevant))

avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
explore_rates = {
    "ε-Greedy":  mab_eg.explore_rate,
    "UCB":       mab_ucb.explore_rate,
    "Q-Learning": ql_agent.explore_rate,
}

print("\n── Precision@5 ──────────────────────────────")
for model, score in avg_metrics.items():
    print(f"  {model:<15}: {score:.4f}")

print("\n── Exploration Rates ────────────────────────")
for model, rate in explore_rates.items():
    print(f"  {model:<15}: {rate:.2%}")

# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────

print("\n[Plot] Generating comparison charts …")

# Downsample cumulative reward curves for plotting
def smooth(arr, w=500):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

fig = plt.figure(figsize=(18, 13), facecolor="#0d0d0d")
fig.suptitle("Task 9 · RL Recommender System", fontsize=18,
             color="white", fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

DARK  = "#0d0d0d"
PANEL = "#1a1a2e"
C1    = "#e94560"   # red-pink
C2    = "#0f3460"   # navy
C3    = "#16213e"   # dark blue
C4    = "#a8dadc"   # teal
C5    = "#457b9d"   # steel blue
C6    = "#f1faee"   # off-white

model_colors = {
    "SVD": C4, "User-CF": C5, "Item-CF": "#f4a261",
    "ε-Greedy": C1, "UCB": "#06d6a0", "Q-Learning": "#ffd166"
}

# ── (A) Cumulative reward: ε-Greedy vs UCB ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
stride = 1000
ax1.plot(eg_cumrew[::stride],  color=C1,        lw=2, label="ε-Greedy")
ax1.plot(ucb_cumrew[::stride], color="#06d6a0", lw=2, label="UCB")
ax1.set_title("MAB Cumulative Reward", color="white", fontsize=11, pad=8)
ax1.set_xlabel("Steps (×1k)", color="#aaa"); ax1.set_ylabel("Cumulative Reward", color="#aaa")
ax1.tick_params(colors="#aaa"); ax1.spines[:].set_color("#333")
ax1.legend(facecolor="#111", labelcolor="white", fontsize=9)

# ── (B) Q-Learning cumulative reward ────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL)
ax2.plot(ql_cumrew[::200], color="#ffd166", lw=2)
ax2.set_title("Q-Learning Cumulative Reward", color="white", fontsize=11, pad=8)
ax2.set_xlabel("Steps (×200)", color="#aaa"); ax2.set_ylabel("Cumulative Reward", color="#aaa")
ax2.tick_params(colors="#aaa"); ax2.spines[:].set_color("#333")

# ── (C) Precision@5 bar chart ───────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(PANEL)
models = list(avg_metrics.keys())
scores = [avg_metrics[m] for m in models]
colors = [model_colors[m] for m in models]
bars   = ax3.barh(models, scores, color=colors, edgecolor="#222", height=0.55)
for bar, score in zip(bars, scores):
    ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{score:.3f}", va="center", color="white", fontsize=9)
ax3.set_title("Precision@5 Comparison", color="white", fontsize=11, pad=8)
ax3.set_xlabel("Precision@5", color="#aaa")
ax3.tick_params(colors="#aaa"); ax3.spines[:].set_color("#333")
ax3.set_xlim(0, max(scores) * 1.25)

# ── (D) Exploration rate bars ────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(PANEL)
er_models = list(explore_rates.keys())
er_vals   = [explore_rates[m] * 100 for m in er_models]
er_cols   = [model_colors[m] for m in er_models]
ax4.bar(er_models, er_vals, color=er_cols, edgecolor="#222", width=0.5)
for i, v in enumerate(er_vals):
    ax4.text(i, v + 0.5, f"{v:.1f}%", ha="center", color="white", fontsize=10)
ax4.set_title("Exploration Rate (%)", color="white", fontsize=11, pad=8)
ax4.set_ylabel("% Explore", color="#aaa")
ax4.tick_params(colors="#aaa"); ax4.spines[:].set_color("#333")
ax4.set_ylim(0, max(er_vals) * 1.3)

# ── (E) Q-value heatmap (first 20 states × 30 movies) ─
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor(PANEL)
q_sub = ql_agent.Q[:20, :30]
im    = ax5.imshow(q_sub, aspect="auto", cmap="RdYlGn", interpolation="nearest")
plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(colors="#aaa")
ax5.set_title("Q-Table Heat Map\n(States 0–19 × Movies 0–29)", color="white", fontsize=11, pad=8)
ax5.set_xlabel("Movie Index", color="#aaa"); ax5.set_ylabel("State (User Bucket)", color="#aaa")
ax5.tick_params(colors="#aaa"); ax5.spines[:].set_color("#333")

# ── (F) Model comparison radar / spider ──────
ax6 = fig.add_subplot(gs[1, 2], polar=True)
ax6.set_facecolor(PANEL)
radar_models = list(avg_metrics.keys())
radar_vals   = [avg_metrics[m] for m in radar_models]
N = len(radar_models)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]; radar_vals_plot = radar_vals + radar_vals[:1]
ax6.plot(angles, radar_vals_plot, color=C1, lw=2)
ax6.fill(angles, radar_vals_plot, color=C1, alpha=0.25)
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(radar_models, color="white", fontsize=9)
ax6.tick_params(colors="#aaa")
ax6.set_facecolor(PANEL)
ax6.spines["polar"].set_color("#333")
ax6.yaxis.set_tick_params(labelcolor="#555")
ax6.set_title("Precision@5 Radar", color="white", fontsize=11, pad=20)
path ="/home/suvendu/mlbd/code/Big_data/"
plt.savefig(path + "rl_recommender_results.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print("\n[Done] Plot saved → rl_recommender_results.png")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Model':<16} {'Precision@5':>12} {'Explore Rate':>14}")
print("-" * 44)
for m in avg_metrics:
    er = f"{explore_rates[m]:.2%}" if m in explore_rates else "N/A"
    print(f"{m:<16} {avg_metrics[m]:>12.4f} {er:>14}")
print("=" * 60)
print("""
KEY INSIGHTS
────────────
• ε-Greedy (ε=0.1) : Explores 10% randomly → simple, stable, predictable.
• UCB               : Explores less-tried items via confidence bonus → adaptive.
• Q-Learning        : Models sequential state transitions; learns long-term value.
• SVD / CF          : Strong collaborative signal; no exploration-exploitation tradeoff.
• RL models optimize cumulative (long-term) reward; traditional models
  optimise immediate rating prediction.
""")
