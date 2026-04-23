import os
import json
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt


ATTENTION_MODELS = {
    "CaST", "GMAN", "STAEformer", "STGNN", "STGraformer", "TGAT"
}

RECURRENT_MODELS = {
    "DCRNN", "DyGrAE", "EvolveGCNO", "EvolveGCNH",
    "GCLSTM", "GConvGRU", "GConvLSTM", "MPNNLSTM", "TGCN"
}

CONVOLUTIONAL_MODELS = {
    "AAGCN", "GraphWaveNet", "LSGCN", "MTGNN", "SLCNN", "STGCN"
}


def get_family(arch):
    if arch in ATTENTION_MODELS:
        return "attention"
    elif arch in RECURRENT_MODELS:
        return "recurrent"
    elif arch in CONVOLUTIONAL_MODELS:
        return "convolutional"

def _process_metrics_file(args):

    dataset, arch, config, seed_path = args

    metrics_path = os.path.join(seed_path, "metrics.json")

    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
    except:
        return None

    hidden_match = re.search(r"hid(\d+)", config)
    hidden = int(hidden_match.group(1)) if hidden_match else None

    row = {
        "dataset": dataset,
        "architecture": arch,
        "family": get_family(arch),
        "config_name": config,
        "hidden": hidden,
        "seed": data["seed"],
        "test_rmse": data["test_rmse"],
        "test_r2_global": data.get("test_r2_global", None),
    }

    for k, v in data["config"].items():
        row[k] = v

    return row


class ResultsAnalyzer:

    def __init__(self, roots, num_workers=None):
        if isinstance(roots, str):
            roots = [roots]
        self.roots = roots
        self.df = None
        self.num_workers = num_workers or os.cpu_count()

    def load(self):

        tasks = []

        for root in self.roots:

            # 🔴 FILTRO PRINCIPAL: só aceita pastas *_long
            if not root.endswith("_long"):
                continue

            dataset = root.replace("results_", "").replace("_long", "")

            if not os.path.isdir(root):
                continue

            for arch in os.listdir(root):

                arch_path = os.path.join(root, arch)
                if not os.path.isdir(arch_path):
                    continue

                for config in os.listdir(arch_path):

                    config_path = os.path.join(arch_path, config)
                    if not os.path.isdir(config_path):
                        continue

                    for seed in os.listdir(config_path):

                        seed_path = os.path.join(config_path, seed)
                        if not os.path.isdir(seed_path):
                            continue

                        tasks.append((dataset, arch, config, seed_path))

        rows = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_metrics_file, t) for t in tasks]

            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    rows.append(r)

        self.df = pd.DataFrame(rows)

        print("\n[LOAD] Shape:", self.df.shape)

        if "test_r2_global" in self.df.columns:

            print("\n[Top 3 architectures per dataset by R²]")

            for dataset in sorted(self.df["dataset"].unique()):

                group = self.df[self.df["dataset"] == dataset]

                best = (
                    group.groupby("architecture")["test_r2_global"]
                    .max()
                    .sort_values(ascending=False)
                    .head(3)
                )

                print(f"\nDataset: {dataset}")
                print(best)

        return self.df

    def plot_cd_subplots(self, output="cd_diagrams_all_datasets_long.pdf"):

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import scikit_posthocs as sp
        from scipy.stats import friedmanchisquare

        os.makedirs("stats_cd", exist_ok=True)

        dataset_names = {
            "chickenpox": "a) Hungary Chickenpox Dataset",
            "wikimaths": "b) Wikipedia Mathematics Dataset",
            "englandcovid": "c) England COVID-19 Dataset",
            "montevideobus": "d) Montevideo Bus Dataset",
        }

        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12
        })

        dataset_order = [
            "chickenpox",
            "wikimaths",
            "englandcovid",
            "montevideobus",
        ]

        datasets = [d for d in dataset_order if d in self.df["dataset"].unique()]

        fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 3 * len(datasets)))

        fig.suptitle("Critical Difference Diagrams (LONG)", fontsize=18, y=0.96)

        if len(datasets) == 1:
            axes = [axes]

        family_colors = {
            "attention": "#D4AF37FF",
            "recurrent": "#525252FF",
            "convolutional": "#CA0324FF"
        }

        for ax, dataset in zip(axes, datasets):

            group = self.df[self.df["dataset"] == dataset]

            pivot = group.pivot_table(
                index=["lags", "horizon", "hidden", "seed"],
                columns="architecture",
                values="test_rmse"
            ).dropna()

            if pivot.shape[1] < 3:
                ax.set_title(f"{dataset} (skip)")
                ax.axis("off")
                continue

            ranks = pivot.rank(axis=1)
            avg_rank = ranks.mean().sort_values()

            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.columns = pivot.columns
            nemenyi.index = pivot.columns

            ax.set_title(dataset_names.get(dataset, dataset), fontsize=16)

            plt.sca(ax)

            sp.critical_difference_diagram(
                ranks=avg_rank,
                sig_matrix=nemenyi,
                label_fmt_left="{label} [{rank:.2f}]  ",
                label_fmt_right="  [{rank:.2f}] {label}",
                text_h_margin=0.3,
                label_props={"fontweight": "bold", "fontsize": 8},
                marker_props={"marker": "o", "s": 30, "edgecolor": "black"},
                elbow_props={"linewidth": 1.6},
                crossbar_props={"color": "black", "linewidth": 2.2},
            )

            archs = list(avg_rank.index)

            def fam_color(a):
                return family_colors[get_family(a)]

            def get_arch_from_x(x):
                return min(archs, key=lambda a: abs(float(avg_rank[a]) - x))

            for t in ax.texts:
                txt = t.get_text()
                for arch in archs:
                    if arch in txt:
                        t.set_color(fam_color(arch))
                        break

            for line in ax.lines:
                xdata = np.asarray(line.get_xdata(), float)
                ydata = np.asarray(line.get_ydata(), float)

                if xdata.size == 0 or ydata.size == 0:
                    continue

                if np.allclose(ydata, ydata[0]):
                    line.set_color("black")
                    line.set_linewidth(2.2)
                    continue

                y_top = np.max(ydata)
                idx = np.where(np.isclose(ydata, y_top))[0]
                if idx.size == 0:
                    continue

                x_top = float(np.mean(xdata[idx]))
                arch = get_arch_from_x(x_top)

                line.set_color(fam_color(arch))
                line.set_linewidth(1.6)

            for coll in ax.collections:
                offsets = coll.get_offsets()
                if offsets is None or len(offsets) == 0:
                    continue

                xs = np.asarray(offsets)[:, 0]
                cols = []

                for x in xs:
                    arch = get_arch_from_x(float(x))
                    cols.append(fam_color(arch))

                coll.set_facecolor(cols)
                coll.set_edgecolor("black")
                coll.set_linewidth(0.8)

            if dataset == "chickenpox":
                for t in ax.texts:
                    if "STGCN" in t.get_text():
                        t.set_color("#CA0324FF")

                target_lines = []
                for line in ax.lines:
                    xdata = np.asarray(line.get_xdata(), float)
                    ydata = np.asarray(line.get_ydata(), float)
                    if xdata.size == 0 or ydata.size == 0:
                        continue
                    if np.allclose(ydata, ydata[0]):
                        continue
                    y_top = np.max(ydata)
                    idx = np.where(np.isclose(ydata, y_top))[0]
                    if idx.size == 0:
                        continue
                    x_top = float(np.mean(xdata[idx]))
                    if np.isclose(x_top, 14.70, atol=0.02):
                        target_lines.append((line, float(np.min(ydata))))

                if len(target_lines) >= 2:
                    target_lines.sort(key=lambda z: z[1], reverse=True)
                    target_lines[0][0].set_color("#525252FF")
                    target_lines[0][0].set_linewidth(1.6)
                    target_lines[1][0].set_color("#CA0324FF")
                    target_lines[1][0].set_linewidth(1.6)
                elif len(target_lines) == 1:
                    target_lines[0][0].set_color("#525252FF")
                    target_lines[0][0].set_linewidth(1.6)

                for coll in ax.collections:
                    offsets = coll.get_offsets()
                    if offsets is None:
                        continue
                    fc = coll.get_facecolors()
                    for i, (x, y) in enumerate(offsets):
                        if np.isclose(x, 14.70, atol=0.02):
                            arch = get_arch_from_x(float(x))
                            if arch == "DyGrAE":
                                fc[i] = np.array([0.32, 0.32, 0.32, 1.0])
                    coll.set_facecolors(fc)

            if dataset == "wikimaths":
                for t in ax.texts:
                    if "STGCN" in t.get_text():
                        t.set_color("#CA0324FF")

            if dataset == "englandcovid":
                for t in ax.texts:
                    if "STGCN" in t.get_text():
                        t.set_color("#CA0324FF")

                stgcn_rank = float(avg_rank["STGCN"]) if "STGCN" in avg_rank.index else None
                gclstm_rank = float(avg_rank["GCLSTM"]) if "GCLSTM" in avg_rank.index else None

                if stgcn_rank is not None:
                    for line in ax.lines:
                        xdata = np.asarray(line.get_xdata(), float)
                        ydata = np.asarray(line.get_ydata(), float)
                        if xdata.size == 0 or ydata.size == 0:
                            continue
                        if np.allclose(ydata, ydata[0]):
                            continue
                        y_top = np.max(ydata)
                        idx = np.where(np.isclose(ydata, y_top))[0]
                        if idx.size == 0:
                            continue
                        x_top = float(np.mean(xdata[idx]))
                        if np.isclose(x_top, stgcn_rank, atol=0.02):
                            line.set_color("#CA0324FF")
                            line.set_linewidth(1.6)

                    for coll in ax.collections:
                        offsets = coll.get_offsets()
                        if offsets is None:
                            continue
                        fc = coll.get_facecolors()
                        for i, (x, y) in enumerate(offsets):
                            if np.isclose(float(x), stgcn_rank, atol=0.02):
                                fc[i] = np.array([0.79, 0.01, 0.14, 1.0])
                        coll.set_facecolors(fc)

                if gclstm_rank is not None:
                    target_lines = []
                    for line in ax.lines:
                        xdata = np.asarray(line.get_xdata(), float)
                        ydata = np.asarray(line.get_ydata(), float)
                        if xdata.size == 0 or ydata.size == 0:
                            continue
                        if np.allclose(ydata, ydata[0]):
                            continue
                        y_top = np.max(ydata)
                        idx = np.where(np.isclose(ydata, y_top))[0]
                        if idx.size == 0:
                            continue
                        x_top = float(np.mean(xdata[idx]))
                        if np.isclose(x_top, gclstm_rank, atol=0.02):
                            target_lines.append((line, float(np.min(ydata))))

                    if len(target_lines) > 0:
                        target_lines.sort(key=lambda z: z[1], reverse=True)
                        target_lines[0][0].set_color("#525252FF")
                        target_lines[0][0].set_linewidth(1.6)

            if dataset == "montevideobus":
                for t in ax.texts:
                    if "STGCN" in t.get_text():
                        t.set_color("#CA0324FF")

            ax.set_rasterized(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output, dpi=400, bbox_inches="tight")
        plt.close()

        print(f"\nSaved CD subplot figure → {output}")