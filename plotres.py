# plot results
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt


RESULTS = Path("work", "_", "results", "results.csv")
OUTDIR = Path("plots")

plt.close("all")
plt.style.use('seaborn-colorblind')

# load results
df = pd.read_csv(RESULTS)

# filters results
df = df[df.FF_REF < 0.8]

# correlations
def correlation(data, figname, ref, est, lims=None):
    fig = plt.figure(figname)
    x = data[ref]
    y = data[est]
    n = len(x)
    plt.scatter(x, y)
    plt.xlabel(ref)
    plt.ylabel(est)
    plt.grid()

    # regression/correlation
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    line = f'n={n}, y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    plt.plot(x, intercept + slope * x, ":", label=line)
    plt.legend()
    plt.title(figname)
    if lims:
        plt.xlim(lims)
        plt.ylim(lims)
    plt.savefig((OUTDIR / figname).with_suffix(".pdf"))


# for branch, group in df.groupby("branch"):
#     name = branch.replace(".", "-")
#     correlation(group, f"WT1 {name}", "WT1_REF", "WT1_EST", lim=[600, 2000])
#     correlation(group, f"FF {name}", "FF_REF", "FF_EST", lim=[0, 1])


# metrics: RMSE, STD, PSNR
def bars(data, figname, columns, groups):
    if isinstance(columns, str):
        columns = [columns]

    fig, axes = plt.subplots(ncols=len(columns), num=figname)
    for ax, col in zip(axes, columns):
        mean = data.groupby(groups)[col].mean()
        barplot = ax.bar([""] * len(mean), mean)
        ax.set_title(col)
        ax.grid("y")
    handles = barplot.patches
    labels = mean.index
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig((OUTDIR / figname).with_suffix(".pdf"))

bars(df, "WT1", ["WT1_RMSE", "WT1_STD", "WT1_PSNR"], "branch")
bars(df, "FF", ["FF_RMSE", "FF_STD", "FF_PSNR"], "branch")

# make outputdir
OUTDIR.mkdir(exist_ok=True)
