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


def get_family(arch):
    if arch in ATTENTION_MODELS:
        return "attention"
    elif arch in RECURRENT_MODELS:
        return "recurrent"
    else:
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

            dataset = root.replace("results_", "")

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


    def plot_cd_subplots(self, output="cd_diagrams_all_datasets.pdf"):

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

        fig.suptitle(
            "Critical Difference Diagrams",
            fontsize=18,
            y=0.96
        )

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

            stat, p = friedmanchisquare(*[pivot[c] for c in pivot.columns])

            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.columns = pivot.columns
            nemenyi.index = pivot.columns

            title = dataset_names.get(dataset, dataset)
            ax.set_title(title, fontsize=16)

            plt.sca(ax)

            sp.critical_difference_diagram(
                ranks=avg_rank,
                sig_matrix=nemenyi,
                label_fmt_left="{label} [{rank:.2f}]  ",
                label_fmt_right="  [{rank:.2f}] {label}",
                text_h_margin=0.3,
                label_props={"fontweight": "bold", "fontsize": 8},
                crossbar_props={"color": "black", "linewidth": 2.2},
                marker_props={"marker": "o", "s": 30, "color": "black", "edgecolor": "black"},
                elbow_props={"color": "black", "linewidth": 1.6},
            )

            archs = list(avg_rank.index)

            color_map = {
                arch: family_colors[get_family(arch)]
                for arch in archs
            }

            # Colore os labels pelo tipo de arquitetura
            for t in ax.texts:
                for arch in archs:
                    if arch in t.get_text():
                        t.set_color(color_map[arch])
                        break

            # Colore os pontos
            for coll in ax.collections:
                offsets = coll.get_offsets()
                if offsets is None or len(offsets) == 0:
                    continue

                xs = np.asarray(offsets)[:, 0]
                cols = [color_map[min(archs, key=lambda a: abs(avg_rank[a] - x))] for x in xs]
                coll.set_facecolor(cols)
                coll.set_edgecolor("black")
                coll.set_linewidth(0.8)

            # Colore SOMENTE os elbows.
            # As linhas de significância estatística permanecem pretas.
            for line in ax.lines:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)

                if xdata.size == 0 or ydata.size == 0:
                    continue

                # Linha horizontal pura = barra de significância estatística
                if np.allclose(ydata, ydata[0]):
                    line.set_color("black")
                    line.set_linewidth(2.2)
                    continue

                # Caso contrário, é elbow/conexão do método
                # Pegamos o x do ponto do elbow como sendo o x mais próximo de algum rank médio
                elbow_x = min(
                    xdata,
                    key=lambda x: min(abs(avg_rank[a] - x) for a in archs)
                )

                closest_arch = min(archs, key=lambda a: abs(avg_rank[a] - elbow_x))
                line.set_color(color_map[closest_arch])
                line.set_linewidth(1.6)

            ax.set_rasterized(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output, dpi=400, bbox_inches="tight")
        plt.close()

        print(f"\nSaved CD subplot figure → {output}")