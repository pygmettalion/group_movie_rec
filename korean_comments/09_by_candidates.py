"""후보 수별 효과 분석 (모델 재학습·조정 없음, 이미 채점된 결과만 다시 나눠 봄).

'평균 최대' 대비 1차 Risk-Aware 의 Strong Dislike 감소를, 그룹당 후보 영화 수 구간별로 계산 -> figures/fig8_by_candidates.png
사용 데이터: 새 시험지 1·2·3번 (1차는 세 시험지 모두를 보기 전에 고정됨).
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, "src")
from group import paired_bootstrap  # noqa: E402

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent
for f in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

ncand = {}
for h, p in [(1, "results_v2/FRESH_HOLDOUT_groups.csv"), (2, "results_v3/FRESH_HOLDOUT2_groups.csv"),
             (3, "results_v4/FRESH_HOLDOUT3_groups.csv")]:
    d = pd.read_csv(ROOT / p)
    ncand.update({(h, g): n for g, n in zip(d.group_id, d.test_common_movies)})
parts = []
a = pd.read_csv(ROOT / "results_v2/FRESH_per_group_p5.csv")
b = pd.read_csv(ROOT / "results_v3/FRESH2_per_group_p5.csv")
c = pd.read_csv(ROOT / "results_v4/FRESH3_per_group.csv").rename(columns={"gid": "group_id"})
for h, d, ours, avg in [(1, a[a.version == "v1"], "RiskAware", "Average"), (2, b[b.version == "v1"], "Final", "Average"),
                        (3, c, "1차 Risk-Aware", "평균 최대")]:
    for m, new in [(ours, "ours"), (avg, "avg")]:
        parts.append(d[d.method == m][["group_id", "avg", "min", "sd"]].assign(h=h, method=new))
df = pd.concat(parts)
df["n"] = [ncand[(h, g)] for h, g in zip(df.h, df.group_id)]
BINS = [("3편", 3, 3), ("4~5편", 4, 5), ("6~9편", 6, 9), ("10편 이상", 10, 999)]
rows = []
for name, lo, hi in BINS:
    s = df[(df.n >= lo) & (df.n <= hi)]
    o = s[s.method == "ours"].sort_values(["h", "group_id"]).reset_index(drop=True)
    v = s[s.method == "avg"].sort_values(["h", "group_id"]).reset_index(drop=True)
    bt = paired_bootstrap(o, v)
    rows.append(dict(bin=name, groups=len(o), avg_sd=v.sd.mean(), ours_sd=o.sd.mean(),
                     reduction=1 - o.sd.mean() / v.sd.mean(), diff=bt["sd"]["diff"], lo=bt["sd"]["lo"], hi=bt["sd"]["hi"],
                     avg_sat_diff=bt["avg"]["diff"]))
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results_v4" / "by_candidates.csv", index=False)
print(t.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(t))
w = 0.36
b1 = ax.bar(x - w / 2, t.avg_sd * 100, w, color="#B4B9C0", label="평균 최대 (최고 기준만 보는 일반 방식)")
b2 = ax.bar(x + w / 2, t.ours_sd * 100, w, color="#2F6FDB", label="우리 모델 (Risk-Aware)")
for i, row in t.iterrows():
    ax.text(x[i] - w / 2, row.avg_sd * 100 + 0.3, f"{row.avg_sd * 100:.1f}%", ha="center", fontsize=10)
    ax.text(x[i] + w / 2, row.ours_sd * 100 + 0.3, f"{row.ours_sd * 100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.text(x[i], max(row.avg_sd, row.ours_sd) * 100 + 2.0, f"Strong Dislike {row.reduction * 100:.0f}% 감소", ha="center",
            fontsize=11, color="#2F6FDB" if row.reduction > 0.2 else "#666666", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"후보 {r.bin}\n({r.groups}그룹)" for r in t.itertuples()])
ax.set_ylabel("4명 중 누군가 Strong Dislike 한 비율 (%)")
ax.set_ylim(0, t[["avg_sd", "ours_sd"]].max().max() * 100 + 5)
ax.set_title("고를 영화가 많을수록 우리 모델이 Strong Dislike 를 더 많이 피한다", fontsize=14, fontweight="bold")
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.01, "새 시험지 1·2·3번 2,000그룹(1차 모델은 모두 보기 전에 고정). 평균 만족도 차이는 모든 구간에서 0.004 이하. "
         "실제 서비스 후보는 수천 편.", fontsize=9, color="#666666")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / "figures" / "fig8_by_candidates.png", dpi=180)
