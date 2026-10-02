import os
import json
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import friedmanchisquare, spearmanr, pearsonr
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

        fig.suptitle("Critical Difference Diagrams (Long)", fontsize=18, y=0.96)

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

                # -------- CORREÇÃO DEFINITIVA DO PONTO --------
                global_left_x = np.inf
                target_point = None

                for coll in ax.collections:
                    offsets = coll.get_offsets()
                    if offsets is None or len(offsets) == 0:
                        continue

                    xs = np.asarray(offsets)[:, 0]
                    idx = np.argmin(xs)

                    if xs[idx] < global_left_x:
                        global_left_x = xs[idx]
                        target_point = (coll, idx, offsets[idx])

                if target_point is not None:
                    coll, idx, (x, y) = target_point

                    # remove o ponto original
                    offsets = coll.get_offsets()
                    new_offsets = np.delete(offsets, idx, axis=0)
                    coll.set_offsets(new_offsets)

                    # recria o ponto corretamente em amarelo
                    ax.scatter(
                        [x], [y],
                        color="#D4AF37FF",
                        edgecolor="black",
                        s=30,
                        zorder=5
                    )

                gman_lines = []
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
                    if np.isclose(x_top, 1.75, atol=0.02):
                        gman_lines.append((line, float(np.min(ydata))))

                if len(gman_lines) >= 2:
                    gman_lines.sort(key=lambda z: z[1], reverse=True)
                    gman_lines[1][0].set_color("#D4AF37FF")
                    gman_lines[1][0].set_linewidth(1.6)

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
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"\nSaved CD subplot figure → {output}")

    def compare_input_projection(self, roots_ip, roots_noip, output_csv="ip_vs_noip.csv"):

        analyzer_ip = ResultsAnalyzer(roots_ip)
        df_ip = analyzer_ip.load()

        analyzer_noip = ResultsAnalyzer(roots_noip)
        df_noip = analyzer_noip.load()

        if df_ip.empty or df_noip.empty:
            raise ValueError("One of the dataframes is empty")

        df_ip = df_ip.copy()
        df_noip = df_noip.copy()

        df_ip["dataset"] = df_ip["dataset"].str.replace("_noipj", "", regex=False)
        df_noip["dataset"] = df_noip["dataset"].str.replace("_noipj", "", regex=False)

        if "montevideobus" in df_noip["dataset"].unique():

            check = (
                df_noip[df_noip["dataset"] == "montevideobus"]
                .groupby("architecture")
                .size()
                .reset_index(name="count")
            )

            valid_arch = check[check["count"] == check["count"].max()]["architecture"].tolist()

            df_noip = df_noip[
                (df_noip["dataset"] != "montevideobus") |
                (df_noip["architecture"].isin(valid_arch))
            ]

            df_ip = df_ip[
                (df_ip["dataset"] != "montevideobus") |
                (df_ip["architecture"].isin(valid_arch))
            ]

        df_ip_grouped = (
            df_ip.groupby(["dataset", "architecture"])["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "rmse_ip", "std": "std_ip"})
        )

        df_noip_grouped = (
            df_noip.groupby(["dataset", "architecture"])["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "rmse_noip", "std": "std_noip"})
        )

        table = pd.merge(
            df_ip_grouped,
            df_noip_grouped,
            on=["dataset", "architecture"],
            how="inner"
        )

        if table.empty:
            raise ValueError("Merge resulted in empty table")

        table["delta"] = table["rmse_noip"] - table["rmse_ip"]
        table["delta_pct"] = table["delta"] / table["rmse_ip"]
        table["family"] = table["architecture"].apply(get_family)

        eps = 0.01

        def impact(x):
            if x > eps:
                return "worse"
            elif x < -eps:
                return "better"
            else:
                return "neutral"

        table["impact"] = table["delta_pct"].apply(impact)

        final = table.copy()
        final.to_csv(output_csv, index=False)

        summary_family = (
            table.groupby(["family", "impact"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        summary_family["total"] = summary_family.sum(axis=1, numeric_only=True)
        summary_family["worse_pct"] = summary_family["worse"] / summary_family["total"] * 100

        print("\n[Impact by family]")
        print(summary_family.sort_values("worse_pct", ascending=False))

        family_colors = {
            "attention": "#D4AF37FF",
            "recurrent": "#525252FF",
            "convolutional": "#CA0324FF"
        }

        family_order = ["attention", "recurrent", "convolutional"]

        dataset_titles = {
            "chickenpox": "Hungary Chickenpox",
            "wikimaths": "Wikipedia Mathematics",
            "englandcovid": "England COVID-19",
            "montevideobus": "Montevideo Bus"
        }

        datasets = sorted(table["dataset"].unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), sharey=True)

        if len(datasets) == 1:
            axes = [axes]

        for i, d in enumerate(datasets):

            sub = table[table["dataset"] == d].copy()

            sub = sub.groupby(["architecture", "family"])["delta"].mean().reset_index()
            sub["family"] = pd.Categorical(sub["family"], categories=family_order, ordered=True)
            sub = sub.sort_values(["family", "architecture"])

            colors = [family_colors[f] for f in sub["family"]]

            x = np.arange(len(sub))

            axes[i].bar(x, sub["delta"], color=colors, width=0.6, align="edge")

            axes[i].axhline(0, color="black", linewidth=1.5)

            axes[i].set_title(dataset_titles.get(d, d), fontsize=18)

            axes[i].set_xticks(x + 0.925)
            axes[i].set_xticklabels(sub["architecture"], rotation=45, ha="right", fontsize=10)

            for tick, fam in zip(axes[i].get_xticklabels(), sub["family"]):
                tick.set_fontweight("bold")
                tick.set_color(family_colors[fam])

            axes[i].grid(axis="y", linestyle="--", alpha=0.4)

            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["left"].set_visible(False)

            axes[i].set_rasterized(True)

        axes[0].set_ylabel("ΔRMSE", fontsize=14)

        fig.suptitle(
            "ΔRMSE by Dataset (No Input Projection vs Input Projection)",
            fontsize=20,
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.subplots_adjust(bottom=0.32, wspace=0.25)

        plt.savefig("ip_delta_rmse_by_dataset.pdf", dpi=200)
        plt.close()

        return final, summary_family
    
    def compare_input_projection(self, roots_ip, roots_noip, output_csv="ip_vs_noip.csv"):

        analyzer_ip = ResultsAnalyzer(roots_ip)
        df_ip = analyzer_ip.load()

        analyzer_noip = ResultsAnalyzer(roots_noip)
        df_noip = analyzer_noip.load()

        if df_ip.empty or df_noip.empty:
            raise ValueError("One of the dataframes is empty")

        df_ip = df_ip.copy()
        df_noip = df_noip.copy()

        df_ip["dataset"] = df_ip["dataset"].str.replace("_noipj", "", regex=False)
        df_noip["dataset"] = df_noip["dataset"].str.replace("_noipj", "", regex=False)

        if "montevideobus" in df_noip["dataset"].unique():

            check = (
                df_noip[df_noip["dataset"] == "montevideobus"]
                .groupby("architecture")
                .size()
                .reset_index(name="count")
            )

            valid_arch = check[
                check["count"] == check["count"].max()
            ]["architecture"].tolist()

            df_noip = df_noip[
                (df_noip["dataset"] != "montevideobus") |
                (df_noip["architecture"].isin(valid_arch))
            ]

            df_ip = df_ip[
                (df_ip["dataset"] != "montevideobus") |
                (df_ip["architecture"].isin(valid_arch))
            ]

        df_ip_grouped = (
            df_ip.groupby(
                ["dataset", "architecture"]
            )["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(
                columns={
                    "mean": "rmse_ip",
                    "std": "std_ip"
                }
            )
        )

        df_noip_grouped = (
            df_noip.groupby(
                ["dataset", "architecture"]
            )["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(
                columns={
                    "mean": "rmse_noip",
                    "std": "std_noip"
                }
            )
        )

        table = pd.merge(
            df_ip_grouped,
            df_noip_grouped,
            on=["dataset", "architecture"],
            how="inner"
        )

        if table.empty:
            raise ValueError("Merge resulted in empty table")

        table["delta"] = (
            table["rmse_noip"] - table["rmse_ip"]
        )

        table["delta_pct"] = (
            table["delta"] / table["rmse_ip"]
        )

        table["family"] = table["architecture"].apply(get_family)

        eps = 0.01

        def impact(x):

            if x > eps:
                return "worse"

            elif x < -eps:
                return "better"

            else:
                return "neutral"

        table["impact"] = table["delta_pct"].apply(impact)

        final = table.copy()

        final.to_csv(output_csv, index=False)

        summary_family = (
            table.groupby(["family", "impact"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        summary_family["total"] = (
            summary_family.sum(axis=1, numeric_only=True)
        )

        summary_family["worse_pct"] = (
            summary_family["worse"] /
            summary_family["total"] * 100
        )

        print("\n[Impact by family]")
        print(
            summary_family.sort_values(
                "worse_pct",
                ascending=False
            )
        )

        family_colors = {
            "attention": "#D4AF37FF",
            "recurrent": "#525252FF",
            "convolutional": "#CA0324FF"
        }

        family_order = [
            "attention",
            "recurrent",
            "convolutional"
        ]

        dataset_titles = {
            "chickenpox": "Hungary Chickenpox",
            "wikimaths": "Wikipedia Mathematics",
            "englandcovid": "England COVID-19",
            "montevideobus": "Montevideo Bus"
        }

        datasets = [
            d for d in [
                "chickenpox",
                "wikimaths",
                "englandcovid",
                "montevideobus"
            ]
            if d in table["dataset"].unique()
        ]

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10),
            sharey=True
        )

        axes = axes.flatten()

        for i, d in enumerate(datasets):

            sub = table[
                table["dataset"] == d
            ].copy()

            sub = (
                sub.groupby(
                    ["architecture", "family"]
                )["delta"]
                .mean()
                .reset_index()
            )

            sub["family"] = pd.Categorical(
                sub["family"],
                categories=family_order,
                ordered=True
            )

            sub = sub.sort_values(
                ["family", "architecture"]
            )

            colors = [
                family_colors[f]
                for f in sub["family"]
            ]

            x = np.arange(len(sub))

            axes[i].bar(
                x,
                sub["delta"],
                color=colors,
                width=0.6,
                align="edge"
            )

            axes[i].axhline(
                0,
                color="black",
                linewidth=1.5
            )

            axes[i].set_title(
                dataset_titles.get(d, d),
                fontsize=18
            )

            axes[i].set_xticks(x + 0.925)

            axes[i].set_xticklabels(
                sub["architecture"],
                rotation=45,
                ha="right",
                fontsize=10
            )

            for tick, fam in zip(
                axes[i].get_xticklabels(),
                sub["family"]
            ):
                tick.set_fontweight("bold")
                tick.set_color(family_colors[fam])

            axes[i].grid(
                axis="y",
                linestyle="--",
                alpha=0.4
            )

            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
            axes[i].spines["left"].set_visible(False)

            axes[i].set_rasterized(True)

        for j in range(len(datasets), 4):
            axes[j].axis("off")

        axes[0].set_ylabel(
            "ΔRMSE",
            fontsize=14
        )

        axes[2].set_ylabel(
            "ΔRMSE",
            fontsize=14
        )

        fig.suptitle(
            "ΔRMSE by Dataset (No Input Projection vs Input Projection)",
            fontsize=20,
            y=0.98
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        plt.subplots_adjust(
            bottom=0.18,
            hspace=0.60,
            wspace=0.18
        )

        plt.savefig(
            "ip_delta_rmse_by_dataset.pdf",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        return final, summary_family

    def spectral_performance_alignment(
        self,
        regime_name="Long-Range",
        output_dir="stats_spectral_alignment",
        output_prefix="long_range",
    ):

        if self.df is None or self.df.empty:
            self.load()

        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Run load() first.")

        os.makedirs(output_dir, exist_ok=True)

        def normalize_dataset_name(dataset):
            dataset = str(dataset).lower()
            dataset = dataset.replace("results_", "")
            dataset = dataset.replace("_long", "")
            dataset = dataset.replace("_noipj", "")

            if dataset == "england":
                dataset = "englandcovid"

            return dataset

        spectral = pd.DataFrame([
            {
                "dataset": "chickenpox",
                "dataset_label": "Chickenpox",
                "E_L": 0.4387,
                "E_M": 0.2785,
                "E_H": 0.2828,
                "E_D": 16.26,
            },
            {
                "dataset": "wikimaths",
                "dataset_label": "WikiMaths",
                "E_L": 0.6562,
                "E_M": 0.1702,
                "E_H": 0.1736,
                "E_D": 629.89,
            },
            {
                "dataset": "englandcovid",
                "dataset_label": "EnglandCOVID",
                "E_L": 0.6370,
                "E_M": 0.1856,
                "E_H": 0.1774,
                "E_D": 70.33,
            },
            {
                "dataset": "montevideobus",
                "dataset_label": "Montevideo Bus",
                "E_L": 0.4267,
                "E_M": 0.2902,
                "E_H": 0.2831,
                "E_D": 582.78,
            },
        ])

        spectral["E_MH"] = spectral["E_M"] + spectral["E_H"]
        spectral["dominant_band"] = spectral[["E_L", "E_M", "E_H"]].idxmax(axis=1)

        spectral["spectral_profile"] = np.where(
            spectral["E_L"] >= 0.55,
            "strong_low_frequency",
            "mixed_balanced"
        )

        spectral["expected_architectural_behavior"] = np.where(
            spectral["spectral_profile"] == "strong_low_frequency",
            "lower_architectural_separation",
            "higher_architectural_separation"
        )

        raw = self.df.copy()
        raw["regime"] = regime_name
        raw["dataset"] = raw["dataset"].apply(normalize_dataset_name)
        raw["family"] = raw["architecture"].apply(get_family)

        raw = raw.dropna(subset=["test_rmse"])
        raw = raw[raw["dataset"].isin(spectral["dataset"])].copy()
        raw = raw.dropna(subset=["family"])

        if raw.empty:
            raise ValueError("No matching datasets found after normalization.")

        normalized_rmse = (
            raw
            .groupby(["regime", "dataset", "architecture", "family"], as_index=False)
            .agg(
                mean_rmse=("test_rmse", "mean"),
                std_rmse=("test_rmse", "std"),
                n_runs=("test_rmse", "size"),
            )
        )

        normalized_rmse["best_rmse_dataset_regime"] = (
            normalized_rmse
            .groupby(["regime", "dataset"])["mean_rmse"]
            .transform("min")
        )

        normalized_rmse["normalized_rmse"] = (
            normalized_rmse["mean_rmse"]
            / normalized_rmse["best_rmse_dataset_regime"]
        )

        normalized_rmse["normalized_rmse_gap"] = (
            normalized_rmse["normalized_rmse"] - 1.0
        )

        normalized_rmse["normalized_rmse_gap_pct"] = (
            100.0 * normalized_rmse["normalized_rmse_gap"]
        )

        normalized_rmse["rank"] = (
            normalized_rmse
            .groupby(["regime", "dataset"])["mean_rmse"]
            .rank(method="average", ascending=True)
        )

        family_rank_mean = (
            normalized_rmse
            .groupby(["regime", "dataset", "family"], as_index=False)
            .agg(
                family_mean_rank=("rank", "mean"),
                family_median_rank=("rank", "median"),
                family_min_rank=("rank", "min"),
                family_mean_normalized_rmse=("normalized_rmse", "mean"),
                family_mean_normalized_rmse_gap_pct=("normalized_rmse_gap_pct", "mean"),
                family_best_normalized_rmse_gap_pct=("normalized_rmse_gap_pct", "min"),
                n_architectures=("architecture", "nunique"),
            )
        )

        performance_dispersion = (
            normalized_rmse
            .groupby(["regime", "dataset"], as_index=False)
            .agg(
                iqr_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    lambda x: x.quantile(0.75) - x.quantile(0.25)
                ),
                range_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    lambda x: x.max() - x.min()
                ),
                std_normalized_rmse_gap_pct=("normalized_rmse_gap_pct", "std"),
                mean_normalized_rmse_gap_pct=("normalized_rmse_gap_pct", "mean"),
                median_normalized_rmse_gap_pct=("normalized_rmse_gap_pct", "median"),
                n_architectures=("architecture", "nunique"),
            )
        )

        performance_dispersion = performance_dispersion.merge(
            spectral,
            on="dataset",
            how="left"
        )

        best_family = (
            family_rank_mean
            .sort_values(["regime", "dataset", "family_mean_rank"])
            .groupby(["regime", "dataset"], as_index=False)
            .first()
            .rename(columns={
                "family": "best_family_by_mean_rank",
                "family_mean_rank": "best_family_mean_rank",
                "family_mean_normalized_rmse_gap_pct": "best_family_mean_normalized_rmse_gap_pct",
            })
        )

        best_architecture = (
            normalized_rmse
            .sort_values(["regime", "dataset", "rank"])
            .groupby(["regime", "dataset"], as_index=False)
            .first()
            .rename(columns={
                "architecture": "best_architecture",
                "family": "best_architecture_family",
                "mean_rmse": "best_architecture_rmse",
                "normalized_rmse": "best_architecture_normalized_rmse",
                "rank": "best_architecture_rank",
            })
        )

        median_iqr = performance_dispersion["iqr_normalized_rmse_gap_pct"].median()

        performance_dispersion["observed_architectural_separation"] = np.where(
            performance_dispersion["iqr_normalized_rmse_gap_pct"] > median_iqr,
            "higher_architectural_separation",
            "lower_architectural_separation"
        )

        performance_dispersion["spectral_architectural_alignment"] = (
            performance_dispersion["observed_architectural_separation"]
            == performance_dispersion["expected_architectural_behavior"]
        )

        spectral_architectural_alignment = (
            performance_dispersion[[
                "regime",
                "dataset",
                "dataset_label",
                "E_L",
                "E_M",
                "E_H",
                "E_MH",
                "E_D",
                "dominant_band",
                "spectral_profile",
                "expected_architectural_behavior",
                "iqr_normalized_rmse_gap_pct",
                "range_normalized_rmse_gap_pct",
                "std_normalized_rmse_gap_pct",
                "observed_architectural_separation",
                "spectral_architectural_alignment",
            ]]
            .merge(
                best_family[[
                    "regime",
                    "dataset",
                    "best_family_by_mean_rank",
                    "best_family_mean_rank",
                    "best_family_mean_normalized_rmse_gap_pct",
                ]],
                on=["regime", "dataset"],
                how="left"
            )
            .merge(
                best_architecture[[
                    "regime",
                    "dataset",
                    "best_architecture",
                    "best_architecture_family",
                    "best_architecture_rmse",
                ]],
                on=["regime", "dataset"],
                how="left"
            )
        )

        normalized_rmse = normalized_rmse.sort_values([
            "regime", "dataset", "rank"
        ])

        family_rank_mean = family_rank_mean.sort_values([
            "regime", "dataset", "family_mean_rank"
        ])

        performance_dispersion = performance_dispersion.sort_values([
            "regime", "dataset"
        ])

        spectral_architectural_alignment = spectral_architectural_alignment.sort_values([
            "regime", "dataset"
        ])

        normalized_rmse.to_csv(
            os.path.join(output_dir, f"{output_prefix}_normalized_rmse.csv"),
            index=False
        )

        family_rank_mean.to_csv(
            os.path.join(output_dir, f"{output_prefix}_family_rank_mean.csv"),
            index=False
        )

        performance_dispersion.to_csv(
            os.path.join(output_dir, f"{output_prefix}_performance_dispersion.csv"),
            index=False
        )

        spectral_architectural_alignment.to_csv(
            os.path.join(output_dir, f"{output_prefix}_spectral_architectural_alignment.csv"),
            index=False
        )

        print("\n[1] Normalized RMSE by dataset/regime")
        print(normalized_rmse)

        print("\n[2] Mean rank by family")
        print(family_rank_mean)

        print("\n[3] Performance dispersion by dataset")
        print(performance_dispersion)

        print("\n[4] Spectral-architectural alignment table")
        print(spectral_architectural_alignment)

        return {
            "normalized_rmse": normalized_rmse,
            "family_rank_mean": family_rank_mean,
            "performance_dispersion": performance_dispersion,
            "spectral_architectural_alignment": spectral_architectural_alignment,
        }