"""
March Madness Prediction using Logistic (Softmax) Regression
=============================================================
Builds team-level features from regular season data, trains a logistic
regression model on historical NCAA tournament outcomes, and produces
submission files in the Kaggle competition format.

Features per team-season:
  - Win rate
  - Average scoring margin
  - Average points scored / allowed
  - Seed (numeric, from tourney seeds file)
  - Strength of schedule proxy (avg opponent win rate)
  - Home/away win rates
  - Overtime game rate

For each matchup the model receives the *difference* in features
between Team A and Team B (lower ID minus higher ID) and predicts
the probability that the lower-ID team wins.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings, sys

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────
DATA = Path("/mnt/user-data/uploads")
OUT  = Path("/mnt/user-data/outputs")
OUT.mkdir(exist_ok=True)

# ── helper: load CSV (handles \r\n) ────────────────────────────────
def load(name):
    return pd.read_csv(DATA / name)

# ══════════════════════════════════════════════════════════════════════
#  1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading data …")

# Men's
m_season   = load("MRegularSeasonCompactResults.csv")
m_tourney  = load("MNCAATourneyCompactResults.csv")
m_seeds    = load("MNCAATourneySeeds.csv")

# Women's
w_season   = load("WRegularSeasonCompactResults.csv")
w_tourney  = load("WNCAATourneyCompactResults.csv")
w_seeds    = load("WNCAATourneySeeds.csv")

# Submissions
sub1 = load("SampleSubmissionStage1.csv")
sub2 = load("SampleSubmissionStage2.csv")

# Tag gender and combine
for df in [m_season, m_tourney, m_seeds]: df["Gender"] = "M"
for df in [w_season, w_tourney, w_seeds]: df["Gender"] = "W"

reg     = pd.concat([m_season, w_season], ignore_index=True)
tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)
seeds   = pd.concat([m_seeds, w_seeds], ignore_index=True)

# ══════════════════════════════════════════════════════════════════════
#  2.  PARSE SEEDS  →  numeric seed per team-season
# ══════════════════════════════════════════════════════════════════════
print("Parsing seeds …")

def parse_seed(s):
    """E.g. 'W01a' → 1, 'X16b' → 16"""
    return int(s[1:3])

seeds["SeedNum"] = seeds["Seed"].apply(parse_seed)
seed_lookup = seeds.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

# ══════════════════════════════════════════════════════════════════════
#  3.  BUILD TEAM-SEASON FEATURES from regular-season results
# ══════════════════════════════════════════════════════════════════════
print("Engineering features …")

def build_team_season_features(reg_df):
    """Return a DataFrame indexed by (Season, TeamID) with features."""
    rows = []

    # --- winning-side records ---
    win = reg_df.groupby(["Season", "WTeamID"]).agg(
        wins      = ("WScore", "count"),
        pts_for_w = ("WScore", "sum"),
        pts_ag_w  = ("LScore", "sum"),
        home_wins = ("WLoc", lambda x: (x == "H").sum()),
        away_wins = ("WLoc", lambda x: (x == "A").sum()),
        ot_w      = ("NumOT", lambda x: (x > 0).sum()),
    ).reset_index().rename(columns={"WTeamID": "TeamID"})

    # --- losing-side records ---
    loss = reg_df.groupby(["Season", "LTeamID"]).agg(
        losses     = ("LScore", "count"),
        pts_for_l  = ("LScore", "sum"),
        pts_ag_l   = ("WScore", "sum"),
        home_losses= ("WLoc", lambda x: (x == "A").sum()),   # winner was Away → loser was Home
        away_losses= ("WLoc", lambda x: (x == "H").sum()),   # winner was Home → loser was Away
        ot_l       = ("NumOT", lambda x: (x > 0).sum()),
    ).reset_index().rename(columns={"LTeamID": "TeamID"})

    merged = pd.merge(win, loss, on=["Season", "TeamID"], how="outer").fillna(0)

    merged["games"]       = merged["wins"] + merged["losses"]
    merged["win_rate"]    = merged["wins"] / merged["games"]
    merged["pts_for"]     = (merged["pts_for_w"] + merged["pts_for_l"]) / merged["games"]
    merged["pts_ag"]      = (merged["pts_ag_w"]  + merged["pts_ag_l"])  / merged["games"]
    merged["margin"]      = merged["pts_for"] - merged["pts_ag"]
    merged["home_win_rt"] = merged["home_wins"] / (merged["home_wins"] + merged["home_losses"]).replace(0, 1)
    merged["away_win_rt"] = merged["away_wins"] / (merged["away_wins"] + merged["away_losses"]).replace(0, 1)
    merged["ot_rate"]     = (merged["ot_w"] + merged["ot_l"]) / merged["games"]

    return merged[["Season", "TeamID", "games", "win_rate", "pts_for",
                    "pts_ag", "margin", "home_win_rt", "away_win_rt", "ot_rate"]]

features = build_team_season_features(reg)

# ── Strength of schedule (average opponent win-rate) ────────────────
print("Computing strength of schedule …")

# Build opponent lists
opp_w = reg[["Season", "WTeamID", "LTeamID"]].rename(columns={"WTeamID": "TeamID", "LTeamID": "OppID"})
opp_l = reg[["Season", "LTeamID", "WTeamID"]].rename(columns={"LTeamID": "TeamID", "WTeamID": "OppID"})
opps  = pd.concat([opp_w, opp_l], ignore_index=True)

wr_map = features.set_index(["Season", "TeamID"])["win_rate"].to_dict()
opps["opp_wr"] = opps.apply(lambda r: wr_map.get((r.Season, r.OppID), 0.5), axis=1)
sos = opps.groupby(["Season", "TeamID"])["opp_wr"].mean().reset_index().rename(columns={"opp_wr": "sos"})

features = features.merge(sos, on=["Season", "TeamID"], how="left")
features["sos"] = features["sos"].fillna(0.5)

# ── Attach seed (0 = not seeded / unknown) ──────────────────────────
features["seed"] = features.apply(
    lambda r: seed_lookup.get((r.Season, r.TeamID), 0), axis=1
)

# For unseeded teams give a default "seed" of 18 (worse than any real seed)
features["seed_adj"] = features["seed"].replace(0, 18)

FEAT_COLS = ["win_rate", "pts_for", "pts_ag", "margin",
             "home_win_rt", "away_win_rt", "ot_rate", "sos", "seed_adj"]

feat_index = features.set_index(["Season", "TeamID"])

# ══════════════════════════════════════════════════════════════════════
#  4.  BUILD TRAINING SET from historical tournament matchups
# ══════════════════════════════════════════════════════════════════════
print("Building training set …")

def make_matchup_features(tourney_df, feat_index, feat_cols):
    """Vectorised: feature-diff (lower-ID minus higher-ID), label=1 if lower won."""
    t = tourney_df.copy()
    t["LoTeam"] = t[["WTeamID", "LTeamID"]].min(axis=1)
    t["HiTeam"] = t[["WTeamID", "LTeamID"]].max(axis=1)
    t["Label"]  = (t["WTeamID"] == t["LoTeam"]).astype(int)

    feat_np = feat_index[feat_cols].copy()

    lo_f = t.merge(feat_np, left_on=["Season", "LoTeam"], right_index=True, how="left")[feat_cols]
    hi_f = t.merge(feat_np, left_on=["Season", "HiTeam"], right_index=True, how="left")[feat_cols]

    diff = lo_f.values - hi_f.values
    labels = t["Label"].values

    # Drop rows with NaN
    mask = ~np.isnan(diff).any(axis=1)
    return diff[mask], labels[mask]

X_train, y_train = make_matchup_features(tourney, feat_index, FEAT_COLS)
print(f"  Training samples: {len(y_train)}  (win-rate of lower-ID team: {y_train.mean():.3f})")

# ══════════════════════════════════════════════════════════════════════
#  5.  TRAIN LOGISTIC REGRESSION  (softmax / logistic)
# ══════════════════════════════════════════════════════════════════════
print("Training logistic regression …")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(
    C=1.0,            # regularisation strength
    max_iter=2000,
    solver="lbfgs",
    random_state=42,
)
model.fit(X_scaled, y_train)

# ── Reduce seed weight by 50% ──────────────────────────────────────
SEED_WEIGHT = 0.5   # 1.0 = full weight, 0.0 = no seed influence
seed_idx = FEAT_COLS.index("seed_adj")
original_seed_coef = model.coef_[0][seed_idx]
model.coef_[0][seed_idx] *= SEED_WEIGHT
print(f"  Seed coefficient: {original_seed_coef:+.4f} → {model.coef_[0][seed_idx]:+.4f}  (×{SEED_WEIGHT})")

train_acc = (model.predict(X_scaled) == y_train).mean()
train_ll  = -np.mean(
    y_train * np.log(np.clip(model.predict_proba(X_scaled)[:, 1], 1e-15, 1)) +
    (1 - y_train) * np.log(np.clip(1 - model.predict_proba(X_scaled)[:, 1], 1e-15, 1))
)
print(f"  Train accuracy : {train_acc:.4f}")
print(f"  Train log-loss : {train_ll:.4f}")

# Print feature importances
coefs = pd.Series(model.coef_[0], index=FEAT_COLS).sort_values(key=abs, ascending=False)
print("\n  Feature coefficients (scaled):")
for feat, coef in coefs.items():
    print(f"    {feat:14s}  {coef:+.4f}")

# ══════════════════════════════════════════════════════════════════════
#  6.  PREDICT SUBMISSIONS
# ══════════════════════════════════════════════════════════════════════
def predict_submission(sub_df, feat_index, feat_cols, scaler, model, label=""):
    """Vectorised: parse submission IDs, compute feature diffs, predict."""
    print(f"\nPredicting {label} ({len(sub_df)} matchups) …")

    # Parse IDs
    parsed = sub_df["ID"].str.split("_", expand=True).astype(int)
    parsed.columns = ["Season", "LoTeam", "HiTeam"]

    # Build a flat numpy array of features keyed by (Season, TeamID)
    feat_np = feat_index[feat_cols].copy()

    # Merge features for lo and hi teams
    lo_feats = parsed.merge(feat_np, left_on=["Season", "LoTeam"],
                            right_index=True, how="left")[feat_cols]
    hi_feats = parsed.merge(feat_np, left_on=["Season", "HiTeam"],
                            right_index=True, how="left")[feat_cols]

    diff = lo_feats.values - hi_feats.values
    has_nan = np.isnan(diff).any(axis=1)
    missing = has_nan.sum()

    # Fill missing rows with zeros (will predict ~0.5)
    diff = np.nan_to_num(diff, nan=0.0)

    diff_scaled = scaler.transform(diff)
    probs = model.predict_proba(diff_scaled)[:, 1]

    # Force truly-missing matchups to exactly 0.5
    probs[has_nan] = 0.5

    if missing:
        print(f"  ⚠ {missing} matchups had missing features → defaulted to 0.50")

    out = sub_df.copy()
    out["Pred"] = probs
    return out

sub1_out = predict_submission(sub1, feat_index, FEAT_COLS, scaler, model, "Stage 1")
sub2_out = predict_submission(sub2, feat_index, FEAT_COLS, scaler, model, "Stage 2")

# ── Save ────────────────────────────────────────────────────────────
sub1_out.to_csv(OUT / "SubmissionStage1.csv", index=False)
sub2_out.to_csv(OUT / "SubmissionStage2.csv", index=False)

print(f"\n✅  Saved  {OUT / 'SubmissionStage1.csv'}")
print(f"✅  Saved  {OUT / 'SubmissionStage2.csv'}")
print("\nDone!")
