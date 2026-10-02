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
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"\nSaved CD subplot figure → {output}")

    def export_rmse_table(self, output_csv="rmse_summary.csv"):

        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Run load() first.")

        # Agrupa por dataset e arquitetura calculando média e std
        grouped = (
            self.df
            .groupby(["dataset", "architecture"])["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Formata como "mean ± std"
        grouped["rmse"] = grouped.apply(
            lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
        )

        # Pivot: linhas = arquitetura, colunas = datasets
        table = grouped.pivot(
            index="architecture",
            columns="dataset",
            values="rmse"
        )

        # Ordena colunas por ordem desejada (se existirem)
        desired_order = ["chickenpox", "wikimaths", "englandcovid", "montevideobus"]
        cols = [c for c in desired_order if c in table.columns]
        table = table[cols]

        # Salva CSV
        table.to_csv(output_csv)

        print(f"Saved RMSE table → {output_csv}")

    def topk_rmse(self, k=3):

        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        print(f"\n[Top {k} architectures per dataset by RMSE]")

        for dataset in sorted(self.df["dataset"].unique()):

            group = self.df[self.df["dataset"] == dataset]

            # média do RMSE por arquitetura
            ranking = (
                group.groupby("architecture")["test_rmse"]
                .mean()
                .sort_values(ascending=True)  # menor = melhor
                .head(k)
            )

            print(f"\nDataset: {dataset}")
            print(ranking)

    def spectral_performance_alignment(
        self,
        regime_name="Short-Mid",
        output_dir="stats_spectral_alignment",
        output_prefix="short_mid",
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