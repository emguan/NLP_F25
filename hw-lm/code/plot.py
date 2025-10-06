# plot_learning_curve.py
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = Path("learning_curve.csv")
rows = []
with csv_path.open() as f:
    r = csv.DictReader(f)
    for d in r:
        rows.append((d["size"], float(d["error_pct"])))

# map size labels to numeric x (1,2,4,8) for spacing
xmap = {"1":1, "times2":2, "times4":4, "times8":8}
rows = [(xmap.get(s, s), e, s) for s,e in rows if s in xmap]
rows.sort()

xs  = [r[0] for r in rows]
ys  = [r[1] for r in rows]
lbl = [r[2] for r in rows]

plt.figure()
plt.plot(xs, ys, marker="o")
for x,y,s in rows:
    plt.annotate(s, (x,y), textcoords="offset points", xytext=(0,6), ha="center")
plt.xscale("log", base=2)   # nice spacing at 1,2,4,8
plt.xticks(xs, lbl)
plt.xlabel("Training size")
plt.ylabel("Dev error rate (%)")
plt.title("Learning curve (add-λ*)")
plt.grid(True, alpha=0.3)
out = Path("learning_curve.png").resolve()
plt.tight_layout(); plt.savefig(out, dpi=150)
print(f"Saved {out}")
