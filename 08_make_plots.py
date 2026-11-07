"""Make figures for the report/presentation -> figures/*.png  (reads only saved results files; no model retraining)

matplotlib: Python plotting library. The Noto Sans CJK KR font is set to display Korean text (default font if unavailable).
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg")   # save to file only, no display
ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
for f in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
                     "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"})
GREY, ACCENT, INK, MUTED = "#B4B9C0", "#2F6FDB", "#222222", "#666666"


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(b.get_height()), ha="center", va="bottom",
                fontsize=10, color=INK)


# ---------------------------------------------------------------- 1. why our model (key figure)
s = pd.read_csv(ROOT / "results_v3" / "why_ours_summary.csv")
names = {"Random (아무거나)": "아무거나", "Popular (인기순)": "인기순", "Top-rated (평점순)": "평점순",
         "Host (한 사람 계정)": "한 사람\n계정", "Average (평균 최대)": "Average\n(평균)", "Ours (Risk-Aware)": "우리 모델",
         "Most Pleasure (최고점 최대)": "Most\nPleasure", "Borda (순위 합)": "Borda", "Multiplicative (곱 최대)": "Multi-\nplicative",
         "Least Misery (최저점 최대)": "Least\nMisery", "AWM (불만 제외 후 평균)": "AWM"}
s["label"] = s.method.map(names)
cols = [ACCENT if "Ours" in m else GREY for m in s.method]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
b = axes[0].bar(s.label, s.sd * 100, color=cols, width=0.62)
bar_labels(axes[0], b, lambda v: f"{v:.1f}%")
axes[0].set_title("4명 중 누군가 Strong Dislike 한 비율 (낮을수록 좋음)")
axes[0].set_ylabel("%")
b = axes[1].bar(s.label, s["min"], color=cols, width=0.62)
bar_labels(axes[1], b, lambda v: f"{v:.3f}")
axes[1].set_title("가장 불만인 사람의 만족도 (높을수록 좋음)")
axes[1].set_ylim(0, 0.7)
for ax in axes:
    ax.tick_params(axis="x", labelsize=8.5)
fig.suptitle("기존 추천·그룹 추천 방식 10가지와 비교 — 처음 보는 1,000개 그룹 (후보 5편 이상)", fontsize=14, fontweight="bold")
fig.text(0.01, 0.01, "MovieLens 1M Test 기간, 봉인된 새 시험지 1·2번. 평균 만족도: 우리 0.783 vs Average 0.785 (차이 없음) vs Least Misery 0.781. AWM 은 예측 점수가 불호 기준선 아래로 거의 안 내려가 0.5% 그룹에서만 작동",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(FIG / "fig1_why_ours.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- 2. Stage 1 individual prediction model comparison
v1c = pd.read_csv(ROOT / "results_v1_frozen" / "stage1b_structure_compare.csv").set_index("structure").cv_rmse
v1a = pd.read_csv(ROOT / "results_v1_frozen" / "stage1_compare.csv").set_index("model").val_rmse
v2c = pd.read_csv(ROOT / "results_v2" / "v2_stage1_structure_compare.csv").set_index("structure").cv_rmse
v3c = pd.read_csv(ROOT / "results_v3" / "v3_stage1_blender_compare.csv").cv_rmse.min()
rows = [("사용자 평균만", v1a["Baseline: 사용자 평균"]), ("Content (장르)", v1c["Content only"]),
        ("편향만 (k=0)", v1c["k=0 편향만(참고)"]), ("MF (SGD)", v1a["MF"]), ("MF (ALS)", v1c["MF only"]),
        ("1차 Hybrid\n(최종)", v1c["H2_bucket"]), ("2차 Hybrid\n(+평론가 등)", v2c.min()), ("3차 앙상블\n(+kNN)", v3c)]
d = pd.DataFrame(rows, columns=["model", "rmse"])
d["stars"] = d.rmse * 4
fig, ax = plt.subplots(figsize=(11, 4.4))
cols = [ACCENT if "최종" in m else GREY for m in d.model]
b = ax.bar(d.model, d.stars, color=cols, width=0.62)
bar_labels(ax, b, lambda v: f"{v:.3f}")
ax.set_ylim(0.8, 1.08)
ax.set_ylabel("예측 오차 (별점 개수, RMSE)")
ax.set_title("개인 취향 예측: Hybrid 가 MF·Content 각각보다 정확 (Validation, 낮을수록 좋음)")
ax.tick_params(axis="x", labelsize=9.5)
fig.text(0.01, 0.01, "y축이 0.8부터 시작함(차이를 보이기 위해). 2·3차는 개인 예측은 좋아졌지만 그룹 추천 성능은 1차와 차이 없음", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "fig2_stage1_models.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- 3. lambda selection (v1)
lam = pd.read_csv(ROOT / "results_v1_frozen" / "stage3_lambda_search.csv")
x = range(len(lam))
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(x, lam.strong_dislike_rate * 100, color=ACCENT, marker="o", lw=2)
axes[0].set_ylabel("Strong Dislike 발생률 (%)")
axes[0].set_title("벌점 λ 를 키우면 Strong Dislike 가 줄고, 너무 크면 만족도가 떨어진다 (Validation 649그룹)")
axes[1].plot(x, lam.avg_satisfaction, color="#777777", marker="o", lw=2)
axes[1].set_ylabel("평균 만족도 (0~1)")
star = int(lam.balanced_score.idxmax())
for ax in axes:
    ax.axvline(star, color=MUTED, ls="--", lw=1)
axes[0].text(star + 0.15, axes[0].get_ylim()[1] * 0.98, f"선택 λ = {lam.lam[star]}", va="top", color=INK)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([f"{v:g}" for v in lam.lam])
axes[1].set_xlabel("λ (Final = 예상 만족도 − λ × Strong Dislike 위험), 간격은 실험한 순서대로")
fig.tight_layout()
fig.savefig(FIG / "fig3_lambda_tradeoff.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- 4. Strong Dislike risk model comparison
r = pd.read_csv(ROOT / "results_v1_frozen" / "stage2_risk_search.csv")
r = r[r.percentile == 5]
best = r.sort_values("logloss").groupby(["structure", "features"]).head(1)
lab = {("group", "F1_pred"): "그룹 분류기\n예측 점수만", ("group", "F2_pred_margin"): "그룹 분류기\n+ 불호 기준선 거리",
       ("group", "F3_all"): "그룹 분류기\n+ 영화 통계", ("indiv", "I_member"): "개인 확률 → 4명 합치기\n(최종)"}
best["label"] = [lab[(a, b_)] for a, b_ in zip(best.structure, best.features)]
best = best.sort_values("auc")
rule_auc = float(r.rule_auc.iloc[0])
fig, ax = plt.subplots(figsize=(9.5, 4.2))
names_ = ["단순 규칙\n(불호 기준선 아래 예측 인원)"] + best.label.tolist()
vals = [rule_auc] + best.auc.tolist()
cols = [GREY] + [ACCENT if "최종" in n else GREY for n in best.label]
b = ax.bar(names_, vals, color=cols, width=0.6)
bar_labels(ax, b, lambda v: f"{v:.3f}")
ax.set_ylim(0.5, 0.88)
ax.set_ylabel("ROC-AUC (높을수록 좋음)")
ax.set_title("'누가 강하게 싫어할까' 위험 모델: 한 명씩 확률을 구해 합치는 방식이 가장 정확")
ax.tick_params(axis="x", labelsize=9)
fig.text(0.01, 0.01, "Validation 기간 가짜 그룹 28,654개, 영화 단위 5-fold 교차검증. y축 0.5 = 찍기 수준", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "fig4_risk_model.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- 5. Strong Dislike threshold tie problem
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from common import load_movielens  # noqa: E402
from risk import user_thresholds  # noqa: E402
_r, _ = load_movielens()
_tr = _r[_r.split == "train"]
pct = [5, 10, 15, 20, 25]
inclusive, fixed = [], []
for p in pct:   # mean, over users in the Train period, of the 'share flagged as Strong Dislike'
    q = _tr.groupby("user_id").rating_norm.quantile(p / 100)          # usual percentile formula
    inclusive.append(float((_tr.rating_norm <= _tr.user_id.map(q)).groupby(_tr.user_id).mean().mean() * 100))
    fixed.append(float(user_thresholds(_tr, p).share.mean() * 100))   # tie-aware formula (actually used)
fig, ax = plt.subplots(figsize=(8, 4))
xi = range(len(pct))
ax.plot(xi, pct, color=MUTED, ls="--", label="목표 (하위 p%)")
ax.plot(xi, inclusive, color=GREY, marker="o", lw=2, label="보통 백분위 공식")
ax.plot(xi, fixed, color=ACCENT, marker="o", lw=2, label="동점을 고려한 공식 (사용)")
ax.set_xticks(list(xi))
ax.set_xticklabels([f"하위 {p}%" for p in pct])
ax.set_ylabel("실제로 Strong Dislike 로 잡힌 비율 (%)")
ax.set_title("별점 동점 때문에 '하위 5%'가 실제로는 14%를 잡던 문제를 고침")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig5_dislike_threshold.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- 6. v1/v2/v3 comparison (test set 2)
v3 = json.load(open(ROOT / "results_v3" / "V3_FINAL_RESULTS.json", encoding="utf-8"))
t = pd.DataFrame(v3["p5"]["table"])
t = t[t.method.isin(["Average", "Final"])]
t["label"] = [f"{v}차 {'평균 최대' if m == 'Average' else 'Risk-Aware'}" for v, m in zip(t.version.str[1], t.method)]
fig, ax = plt.subplots(figsize=(9, 4))
cols = [ACCENT if lb == "1차 Risk-Aware" else GREY for lb in t.label]
b = ax.bar(t.label, t.strong_dislike_rate * 100, color=cols, width=0.6)
bar_labels(ax, b, lambda v: f"{v:.1f}%")
ax.set_ylabel("Strong Dislike 발생률 (%)")
ax.set_title("2·3차는 1차를 넘지 못함 (2번 새 시험지 500그룹, 차이는 모두 우연 범위)")
fig.tight_layout()
fig.savefig(FIG / "fig6_versions.png", dpi=180)
plt.close(fig)
print("저장:", sorted(p.name for p in FIG.glob("*.png")))

# ---------------------------------------------------------------- 7. v4 final (fresh test set 3)
v4 = json.load(open(ROOT / "results_v4" / "V4_FINAL_RESULTS.json", encoding="utf-8"))
s4 = pd.DataFrame(v4["summary"])
order4 = ["아무거나", "인기순", "평점순", "한 사람 계정", "Least Misery", "평균 최대", "1차 Risk-Aware", "4차 (최종 후보)"]
s4 = s4.set_index("method").loc[order4].reset_index()
lbl = {"4차 (최종 후보)": "4차\n(추가 실험)", "1차 Risk-Aware": "1차\nRisk-Aware", "평균 최대": "평균 최대\n(최고 기준)",
       "한 사람 계정": "한 사람\n계정 추천", "Least Misery": "Least\nMisery", "아무거나": "아무거나", "인기순": "인기순", "평점순": "평점순"}
cols = [ACCENT if m.startswith("4차") else GREY for m in s4.method]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
b = axes[0].bar([lbl[m] for m in s4.method], s4.sd * 100, color=cols, width=0.62)
bar_labels(axes[0], b, lambda v: f"{v:.1f}%")
axes[0].set_title("누군가 Strong Dislike 한 비율 (낮을수록 좋음)")
b = axes[1].bar([lbl[m] for m in s4.method], s4.avg, color=cols, width=0.62)
bar_labels(axes[1], b, lambda v: f"{v:.3f}")
axes[1].set_ylim(0.6, 0.8)
axes[1].set_title("평균 만족도 (높을수록 좋음, y축 0.6부터)")
for ax in axes:
    ax.tick_params(axis="x", labelsize=9)
fig.suptitle("4차(추가 실험, 최종 모델 아님): 개인 맞춤 방식 중 Strong Dislike 가 가장 적고 평균 만족도는 유지 — 3번 시험지 1,000그룹", fontsize=13, fontweight="bold")
fig.text(0.01, 0.01, "Final = 평균 + 0.5×최저 − 3.0×위험. 후보 3편뿐인 이 시험지에선 인기순이 Strong Dislike 는 더 적지만(12.6%) 평균 만족도가 크게 낮음(0.751). 1·2번 시험지에선 인기순 9.9% vs 우리 5.4%",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(FIG / "fig7_v4_final.png", dpi=180)
plt.close(fig)
print("fig7 저장")
