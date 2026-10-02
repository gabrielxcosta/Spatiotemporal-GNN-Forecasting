import os
import json
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import friedmanchisquare, wilcoxon
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


def get_family(architecture):
    if architecture in ATTENTION_MODELS:
        return "attention"
    if architecture in RECURRENT_MODELS:
        return "recurrent"
    if architecture in CONVOLUTIONAL_MODELS:
        return "convolutional"
    return None


def _is_long_root(root):
    root_name = os.path.basename(os.path.normpath(root))
    root_name = root_name.replace("_noipj", "")
    return root_name.endswith("_long")


def _dataset_from_root(root):
    dataset = os.path.basename(os.path.normpath(root))
    dataset = dataset.replace("results_", "")
    dataset = dataset.replace("_noipj", "")
    dataset = dataset.replace("_long", "")
    if dataset == "england":
        dataset = "englandcovid"
    return dataset


def _process_metrics_file(args):
    dataset, architecture, config_name, seed_path = args
    metrics_path = os.path.join(seed_path, "metrics.json")

    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, "r") as file:
            data = json.load(file)
    except Exception:
        return None

    hidden_match = re.search(r"hid(\d+)", str(config_name))
    hidden = int(hidden_match.group(1)) if hidden_match else None

    r2_per_horizon = data.get("test_r2_per_horizon", [])
    if not isinstance(r2_per_horizon, (list, tuple, np.ndarray)):
        r2_per_horizon = []

    row = {
        "dataset": dataset,
        "architecture": architecture,
        "family": get_family(architecture),
        "config_name": config_name,
        "hidden": hidden,
        "seed": data.get("seed", None),
        "test_mse": data.get("test_mse", None),
        "test_rmse": data.get("test_rmse", None),
        "test_mae": data.get("test_mae", None),
        "test_mape": data.get("test_mape", None),
        "test_r2_global": data.get("test_r2_global", None),
        "test_r2_mean_horizon": data.get("test_r2_mean_horizon", None),
        "test_r2_per_horizon": list(r2_per_horizon),
        "runtime_sec": data.get("runtime_sec", None),
        "epochs_ran": data.get("epochs_ran", None),
    }

    for key, value in data.get("config", {}).items():
        row[key] = value

    return row


class ResultsAnalyzer:

    def __init__(self, roots, num_workers=None):
        if isinstance(roots, str):
            roots = [roots]
        self.roots = list(roots)
        self.df = None
        self.num_workers = num_workers or os.cpu_count()

    def load(self):
        tasks = []

        for root in self.roots:
            if not _is_long_root(root):
                continue

            if not os.path.isdir(root):
                continue

            dataset = _dataset_from_root(root)

            for architecture in os.listdir(root):
                architecture_path = os.path.join(root, architecture)

                if not os.path.isdir(architecture_path):
                    continue

                for config_name in os.listdir(architecture_path):
                    config_path = os.path.join(architecture_path, config_name)

                    if not os.path.isdir(config_path):
                        continue

                    for seed_name in os.listdir(config_path):
                        seed_path = os.path.join(config_path, seed_name)

                        if not os.path.isdir(seed_path):
                            continue

                        tasks.append(
                            (dataset, architecture, config_name, seed_path)
                        )

        rows = []

        if tasks:
            with ProcessPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = [
                    executor.submit(_process_metrics_file, task)
                    for task in tasks
                ]

                for future in as_completed(futures):
                    row = future.result()
                    if row is not None:
                        rows.append(row)

        self.df = pd.DataFrame(rows)

        print("\n[LOAD] Shape:", self.df.shape)

        if self.df.empty:
            return self.df

        if "test_r2_global" in self.df.columns:
            print("\n[Top 3 architectures per dataset by R²]")

            for dataset in sorted(self.df["dataset"].dropna().unique()):
                group = self.df[self.df["dataset"] == dataset]

                top = (
                    group
                    .groupby("architecture")["test_r2_global"]
                    .max()
                    .sort_values(ascending=False)
                    .head(3)
                )

                print(f"\nDataset: {dataset}")
                print(top)

        return self.df

    def plot_cd_subplots(
        self,
        output="stats_cd/long_range_cd_diagrams_all_datasets.pdf",
    ):
        if self.df is None or self.df.empty:
            self.load()

        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Run load() first.")

        output_directory = os.path.dirname(output)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        dataset_names = {
            "chickenpox": "a) Hungary Chickenpox Dataset",
            "wikimaths": "b) Wikipedia Mathematics Dataset",
            "englandcovid": "c) England COVID-19 Dataset",
            "montevideobus": "d) Montevideo Bus Dataset",
        }

        dataset_order = [
            "chickenpox",
            "wikimaths",
            "englandcovid",
            "montevideobus",
        ]

        datasets = [
            dataset
            for dataset in dataset_order
            if dataset in self.df["dataset"].unique()
        ]

        if not datasets:
            raise ValueError("No configured datasets were found.")

        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
            }
        )

        fig, axes = plt.subplots(
            len(datasets),
            1,
            figsize=(16, 3.1 * len(datasets)),
        )

        if len(datasets) == 1:
            axes = [axes]

        fig.suptitle(
            "Critical Difference Diagrams (Long-Range)",
            fontsize=18,
            y=0.985,
        )

        family_colors = {
            "attention": "#D4AF37",
            "recurrent": "#525252",
            "convolutional": "#CA0324",
        }

        for axis, dataset in zip(axes, datasets):
            group = self.df[self.df["dataset"] == dataset].copy()

            pivot = group.pivot_table(
                index=["lags", "horizon", "hidden", "seed"],
                columns="architecture",
                values="test_rmse",
                aggfunc="mean",
            ).dropna()

            if pivot.shape[1] < 3 or pivot.empty:
                axis.set_title(f"{dataset} (insufficient paired configurations)")
                axis.axis("off")
                continue

            ranks = pivot.rank(axis=1, ascending=True)
            average_rank = ranks.mean().sort_values()

            statistic, p_value = friedmanchisquare(
                *[pivot[column] for column in pivot.columns]
            )

            nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
            nemenyi.columns = pivot.columns
            nemenyi.index = pivot.columns

            axis.set_title(dataset_names.get(dataset, dataset), fontsize=16)
            plt.sca(axis)

            sp.critical_difference_diagram(
                ranks=average_rank,
                sig_matrix=nemenyi,
                label_fmt_left="{label} [{rank:.2f}]  ",
                label_fmt_right="  [{rank:.2f}] {label}",
                text_h_margin=0.3,
                label_props={"fontweight": "bold", "fontsize": 8},
                crossbar_props={"color": "black", "linewidth": 2.2},
                marker_props={
                    "marker": "o",
                    "s": 30,
                    "color": "black",
                    "edgecolor": "black",
                },
                elbow_props={"color": "black", "linewidth": 1.6},
            )

            architectures = list(average_rank.index)
            color_map = {
                architecture: family_colors.get(
                    get_family(architecture),
                    "black",
                )
                for architecture in architectures
            }

            for text in axis.texts:
                for architecture in architectures:
                    if architecture in text.get_text():
                        text.set_color(color_map[architecture])
                        break

            for collection in axis.collections:
                offsets = collection.get_offsets()

                if offsets is None or len(offsets) == 0:
                    continue

                x_values = np.asarray(offsets)[:, 0]

                colors = [
                    color_map[
                        min(
                            architectures,
                            key=lambda architecture: abs(
                                average_rank[architecture] - x
                            ),
                        )
                    ]
                    for x in x_values
                ]

                collection.set_facecolor(colors)
                collection.set_edgecolor("black")
                collection.set_linewidth(0.8)

            for line in axis.lines:
                x_data = np.asarray(line.get_xdata(), dtype=float)
                y_data = np.asarray(line.get_ydata(), dtype=float)

                if x_data.size == 0 or y_data.size == 0:
                    continue

                if np.allclose(y_data, y_data[0]):
                    line.set_color("black")
                    line.set_linewidth(2.2)
                    continue

                elbow_x = min(
                    x_data,
                    key=lambda x: min(
                        abs(average_rank[architecture] - x)
                        for architecture in architectures
                    ),
                )

                closest_architecture = min(
                    architectures,
                    key=lambda architecture: abs(
                        average_rank[architecture] - elbow_x
                    ),
                )

                line.set_color(color_map[closest_architecture])
                line.set_linewidth(1.6)

            axis.set_rasterized(True)

            print(
                f"\n[CD] {dataset}: "
                f"Friedman statistic = {statistic:.6f}, "
                f"p-value = {p_value:.6e}"
            )

        plt.tight_layout(rect=[0, 0, 1, 0.965])
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"\nSaved CD subplot figure → {output}")

    def export_rmse_table(
        self,
        output_csv="stats_tables/long_range_rmse_summary.csv",
    ):
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Run load() first.")

        output_directory = os.path.dirname(output_csv)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        grouped = (
            self.df
            .groupby(["dataset", "architecture"])["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )

        grouped["rmse"] = grouped.apply(
            lambda row: (
                f"{row['mean']:.4f} ± {row['std']:.4f}"
                if pd.notna(row["std"])
                else f"{row['mean']:.4f} ± 0.0000"
            ),
            axis=1,
        )

        table = grouped.pivot(
            index="architecture",
            columns="dataset",
            values="rmse",
        )

        desired_order = [
            "chickenpox",
            "wikimaths",
            "englandcovid",
            "montevideobus",
        ]

        columns = [
            column
            for column in desired_order
            if column in table.columns
        ]

        table = table[columns]
        table.to_csv(output_csv)

        print(f"Saved RMSE table → {output_csv}")

        return table

    def topk_rmse(self, k=3):
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Run load() first.")

        print(f"\n[Top {k} architectures per dataset by RMSE]")

        result = {}

        for dataset in sorted(self.df["dataset"].unique()):
            ranking = (
                self.df[self.df["dataset"] == dataset]
                .groupby("architecture")["test_rmse"]
                .mean()
                .sort_values(ascending=True)
                .head(k)
            )

            result[dataset] = ranking

            print(f"\nDataset: {dataset}")
            print(ranking)

        return result

    def compare_input_projection(
        self,
        roots_ip,
        roots_noip,
        output_csv="stats_input_projection/long_range_ip_vs_noip.csv",
        output_figure="stats_input_projection/long_range_ip_delta_rmse.pdf",
    ):
        analyzer_ip = ResultsAnalyzer(roots_ip)
        analyzer_noip = ResultsAnalyzer(roots_noip)

        dataframe_ip = analyzer_ip.load()
        dataframe_noip = analyzer_noip.load()

        if dataframe_ip.empty or dataframe_noip.empty:
            raise ValueError("One of the input-projection DataFrames is empty.")

        dataframe_ip = dataframe_ip.copy()
        dataframe_noip = dataframe_noip.copy()

        dataframe_ip["dataset"] = dataframe_ip["dataset"].str.replace(
            "_noipj",
            "",
            regex=False,
        )

        dataframe_noip["dataset"] = dataframe_noip["dataset"].str.replace(
            "_noipj",
            "",
            regex=False,
        )

        merge_columns = [
            "dataset",
            "architecture",
            "lags",
            "horizon",
            "hidden",
            "seed",
        ]

        dataframe_ip["hidden"] = dataframe_ip["hidden"].fillna(-1)
        dataframe_noip["hidden"] = dataframe_noip["hidden"].fillna(-1)

        ip_grouped = (
            dataframe_ip
            .groupby(merge_columns, as_index=False)["test_rmse"]
            .mean()
            .rename(columns={"test_rmse": "rmse_ip"})
        )

        noip_grouped = (
            dataframe_noip
            .groupby(merge_columns, as_index=False)["test_rmse"]
            .mean()
            .rename(columns={"test_rmse": "rmse_noip"})
        )

        paired = ip_grouped.merge(
            noip_grouped,
            on=merge_columns,
            how="inner",
        )

        if paired.empty:
            raise ValueError("No paired IP/NoIP runs were found.")

        paired["delta_rmse"] = (
            paired["rmse_noip"] - paired["rmse_ip"]
        )

        paired["delta_rmse_pct"] = (
            100.0
            * paired["delta_rmse"]
            / paired["rmse_ip"].replace(0, np.nan)
        )

        paired["family"] = paired["architecture"].apply(get_family)

        architecture_summary = (
            paired
            .groupby(
                ["dataset", "architecture", "family"],
                as_index=False,
            )
            .agg(
                mean_delta_rmse=("delta_rmse", "mean"),
                std_delta_rmse=("delta_rmse", "std"),
                median_delta_rmse=("delta_rmse", "median"),
                mean_delta_rmse_pct=("delta_rmse_pct", "mean"),
                n_paired_runs=("delta_rmse", "size"),
                worse_without_ip_rate=(
                    "delta_rmse",
                    lambda values: np.mean(values > 0),
                ),
            )
        )

        architecture_summary["ci95_delta_rmse"] = (
            1.96
            * architecture_summary["std_delta_rmse"].fillna(0.0)
            / np.sqrt(architecture_summary["n_paired_runs"])
        )

        family_summary = (
            architecture_summary
            .groupby("family", as_index=False)
            .agg(
                mean_delta_rmse=("mean_delta_rmse", "mean"),
                median_delta_rmse=("median_delta_rmse", "median"),
                mean_delta_rmse_pct=("mean_delta_rmse_pct", "mean"),
                n_architectures=("architecture", "nunique"),
                worse_without_ip_rate=(
                    "worse_without_ip_rate",
                    "mean",
                ),
            )
        )

        output_directory = os.path.dirname(output_csv)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        architecture_summary.to_csv(output_csv, index=False)

        figure_directory = os.path.dirname(output_figure)
        if figure_directory:
            os.makedirs(figure_directory, exist_ok=True)

        family_order = [
            "attention",
            "recurrent",
            "convolutional",
        ]

        family_colors = {
            "attention": "#D4AF37",
            "recurrent": "#525252",
            "convolutional": "#CA0324",
        }

        dataset_titles = {
            "chickenpox": "Hungary Chickenpox",
            "wikimaths": "Wikipedia Mathematics",
            "englandcovid": "England COVID-19",
            "montevideobus": "Montevideo Bus",
        }

        datasets = [
            dataset
            for dataset in [
                "chickenpox",
                "wikimaths",
                "englandcovid",
                "montevideobus",
            ]
            if dataset in architecture_summary["dataset"].unique()
        ]

        figure, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10),
            sharey=True,
        )

        axes = axes.flatten()

        for axis, dataset in zip(axes, datasets):
            panel = architecture_summary[
                architecture_summary["dataset"] == dataset
            ].copy()

            panel["family"] = pd.Categorical(
                panel["family"],
                categories=family_order,
                ordered=True,
            )

            panel = panel.sort_values(["family", "architecture"])
            positions = np.arange(len(panel))
            colors = [
                family_colors[family]
                for family in panel["family"]
            ]

            axis.bar(
                positions,
                panel["mean_delta_rmse"],
                color=colors,
                width=0.7,
            )

            axis.axhline(0.0, color="black", linewidth=1.2)
            axis.set_title(dataset_titles.get(dataset, dataset), fontsize=15)
            axis.set_xticks(positions)
            axis.set_xticklabels(
                panel["architecture"],
                rotation=45,
                ha="right",
                fontsize=9,
            )

            for tick, family in zip(
                axis.get_xticklabels(),
                panel["family"],
            ):
                tick.set_color(family_colors[family])
                tick.set_fontweight("bold")

            axis.grid(axis="y", linestyle="--", alpha=0.35)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        for axis in axes[len(datasets):]:
            axis.axis("off")

        axes[0].set_ylabel("Mean ΔRMSE")
        axes[2].set_ylabel("Mean ΔRMSE")

        figure.suptitle(
            "Long-Range ΔRMSE: No Input Projection − Input Projection",
            fontsize=18,
            y=0.985,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.955])
        figure.savefig(output_figure, dpi=220, bbox_inches="tight")
        plt.close(figure)

        print("\n[Input Projection: Architecture Summary]")
        print(architecture_summary.to_string(index=False))

        print("\n[Input Projection: Family Summary]")
        print(family_summary.to_string(index=False))

        return paired, architecture_summary, family_summary

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

        spectral = pd.DataFrame(
            [
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
            ]
        )

        spectral["E_MH"] = spectral["E_M"] + spectral["E_H"]
        spectral["dominant_band"] = spectral[
            ["E_L", "E_M", "E_H"]
        ].idxmax(axis=1)

        spectral["spectral_profile"] = np.where(
            spectral["E_L"] >= 0.55,
            "strong_low_frequency",
            "mixed_balanced",
        )

        spectral["expected_architectural_behavior"] = np.where(
            spectral["spectral_profile"] == "strong_low_frequency",
            "lower_architectural_separation",
            "higher_architectural_separation",
        )

        raw = self.df.copy()
        raw["regime"] = regime_name
        raw["dataset"] = raw["dataset"].apply(
            lambda dataset: _dataset_from_root(f"results_{dataset}")
        )
        raw["family"] = raw["architecture"].apply(get_family)

        raw = raw.dropna(subset=["test_rmse", "family"])
        raw = raw[raw["dataset"].isin(spectral["dataset"])].copy()

        if raw.empty:
            raise ValueError("No matching datasets found after normalization.")

        normalized_rmse = (
            raw
            .groupby(
                ["regime", "dataset", "architecture", "family"],
                as_index=False,
            )
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
            .groupby(
                ["regime", "dataset", "family"],
                as_index=False,
            )
            .agg(
                family_mean_rank=("rank", "mean"),
                family_median_rank=("rank", "median"),
                family_min_rank=("rank", "min"),
                family_mean_normalized_rmse=(
                    "normalized_rmse",
                    "mean",
                ),
                family_mean_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    "mean",
                ),
                family_best_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    "min",
                ),
                n_architectures=("architecture", "nunique"),
            )
        )

        performance_dispersion = (
            normalized_rmse
            .groupby(["regime", "dataset"], as_index=False)
            .agg(
                iqr_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    lambda values: (
                        values.quantile(0.75) - values.quantile(0.25)
                    ),
                ),
                range_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    lambda values: values.max() - values.min(),
                ),
                std_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    "std",
                ),
                mean_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    "mean",
                ),
                median_normalized_rmse_gap_pct=(
                    "normalized_rmse_gap_pct",
                    "median",
                ),
                n_architectures=("architecture", "nunique"),
            )
        )

        performance_dispersion = performance_dispersion.merge(
            spectral,
            on="dataset",
            how="left",
        )

        best_family = (
            family_rank_mean
            .sort_values(
                ["regime", "dataset", "family_mean_rank"]
            )
            .groupby(["regime", "dataset"], as_index=False)
            .first()
            .rename(
                columns={
                    "family": "best_family_by_mean_rank",
                    "family_mean_rank": "best_family_mean_rank",
                    "family_mean_normalized_rmse_gap_pct": (
                        "best_family_mean_normalized_rmse_gap_pct"
                    ),
                }
            )
        )

        best_architecture = (
            normalized_rmse
            .sort_values(["regime", "dataset", "rank"])
            .groupby(["regime", "dataset"], as_index=False)
            .first()
            .rename(
                columns={
                    "architecture": "best_architecture",
                    "family": "best_architecture_family",
                    "mean_rmse": "best_architecture_rmse",
                    "normalized_rmse": (
                        "best_architecture_normalized_rmse"
                    ),
                    "rank": "best_architecture_rank",
                }
            )
        )

        median_iqr = performance_dispersion[
            "iqr_normalized_rmse_gap_pct"
        ].median()

        performance_dispersion[
            "observed_architectural_separation"
        ] = np.where(
            performance_dispersion[
                "iqr_normalized_rmse_gap_pct"
            ] > median_iqr,
            "higher_architectural_separation",
            "lower_architectural_separation",
        )

        performance_dispersion[
            "spectral_architectural_alignment"
        ] = (
            performance_dispersion[
                "observed_architectural_separation"
            ]
            == performance_dispersion[
                "expected_architectural_behavior"
            ]
        )

        spectral_architectural_alignment = (
            performance_dispersion[
                [
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
                ]
            ]
            .merge(
                best_family[
                    [
                        "regime",
                        "dataset",
                        "best_family_by_mean_rank",
                        "best_family_mean_rank",
                        "best_family_mean_normalized_rmse_gap_pct",
                    ]
                ],
                on=["regime", "dataset"],
                how="left",
            )
            .merge(
                best_architecture[
                    [
                        "regime",
                        "dataset",
                        "best_architecture",
                        "best_architecture_family",
                        "best_architecture_rmse",
                    ]
                ],
                on=["regime", "dataset"],
                how="left",
            )
        )

        normalized_rmse = normalized_rmse.sort_values(
            ["regime", "dataset", "rank"]
        )

        family_rank_mean = family_rank_mean.sort_values(
            ["regime", "dataset", "family_mean_rank"]
        )

        performance_dispersion = performance_dispersion.sort_values(
            ["regime", "dataset"]
        )

        spectral_architectural_alignment = (
            spectral_architectural_alignment
            .sort_values(["regime", "dataset"])
        )

        normalized_rmse.to_csv(
            os.path.join(
                output_dir,
                f"{output_prefix}_normalized_rmse.csv",
            ),
            index=False,
        )

        family_rank_mean.to_csv(
            os.path.join(
                output_dir,
                f"{output_prefix}_family_rank_mean.csv",
            ),
            index=False,
        )

        performance_dispersion.to_csv(
            os.path.join(
                output_dir,
                f"{output_prefix}_performance_dispersion.csv",
            ),
            index=False,
        )

        spectral_architectural_alignment.to_csv(
            os.path.join(
                output_dir,
                f"{output_prefix}_spectral_architectural_alignment.csv",
            ),
            index=False,
        )

        print("\n[1] Normalized RMSE by dataset/regime")
        print(normalized_rmse.to_string(index=False))

        print("\n[2] Mean rank by family")
        print(family_rank_mean.to_string(index=False))

        print("\n[3] Performance dispersion by dataset")
        print(performance_dispersion.to_string(index=False))

        print("\n[4] Spectral-architectural alignment table")
        print(spectral_architectural_alignment.to_string(index=False))

        return {
            "normalized_rmse": normalized_rmse,
            "family_rank_mean": family_rank_mean,
            "performance_dispersion": performance_dispersion,
            "spectral_architectural_alignment": (
                spectral_architectural_alignment
            ),
        }

    def _ensure_context_horizon_data(self):
        if self.df is None or self.df.empty:
            self.load()

        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Run load() first.")

        required_columns = {
            "dataset",
            "architecture",
            "family",
            "seed",
            "lags",
            "horizon",
            "test_rmse",
        }

        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        base = self.df.copy()

        if "hidden" not in base.columns:
            base["hidden"] = np.nan

        numeric_columns = [
            "seed",
            "lags",
            "horizon",
            "hidden",
            "test_rmse",
            "test_r2_global",
            "test_r2_mean_horizon",
        ]

        for column in numeric_columns:
            if column in base.columns:
                base[column] = pd.to_numeric(
                    base[column],
                    errors="coerce",
                )

        base["family"] = base["family"].fillna(
            base["architecture"].apply(get_family)
        )

        base = base[
            base["family"].isin(
                [
                    "attention",
                    "recurrent",
                    "convolutional",
                ]
            )
        ].copy()

        base = base.dropna(
            subset=[
                "dataset",
                "architecture",
                "family",
                "seed",
                "lags",
                "horizon",
                "test_rmse",
            ]
        ).copy()

        base["seed"] = base["seed"].astype(int)
        base["lags"] = base["lags"].astype(int)
        base["horizon"] = base["horizon"].astype(int)
        base["_hidden_key"] = (
            base["hidden"]
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        return base

    @staticmethod
    def _bh_adjust(p_values):
        p_values = np.asarray(p_values, dtype=float)
        adjusted = np.full(
            p_values.shape,
            np.nan,
            dtype=float,
        )

        valid = np.isfinite(p_values)

        if valid.sum() == 0:
            return adjusted

        p_values_valid = p_values[valid]
        order = np.argsort(p_values_valid)
        ranked = p_values_valid[order]
        count = len(ranked)

        corrected = ranked * count / np.arange(1, count + 1)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]
        corrected = np.clip(corrected, 0.0, 1.0)

        restored = np.empty_like(corrected)
        restored[order] = corrected
        adjusted[valid] = restored

        return adjusted

    @staticmethod
    def _family_replicates(
        frame,
        value_column,
        condition_columns,
    ):
        architecture_replicates = (
            frame
            .groupby(
                condition_columns
                + [
                    "architecture",
                    "family",
                    "seed",
                    "_hidden_key",
                ],
                as_index=False,
            )[value_column]
            .mean()
            .rename(
                columns={
                    value_column: f"architecture_{value_column}"
                }
            )
        )

        family_replicates = (
            architecture_replicates
            .groupby(
                condition_columns
                + [
                    "family",
                    "seed",
                    "_hidden_key",
                ],
                as_index=False,
            )[f"architecture_{value_column}"]
            .mean()
            .rename(
                columns={
                    f"architecture_{value_column}": value_column
                }
            )
        )

        return architecture_replicates, family_replicates

    @staticmethod
    def _family_summary(
        family_replicates,
        value_column,
        condition_columns,
    ):
        summary = (
            family_replicates
            .groupby(
                condition_columns + ["family"],
                as_index=False,
            )
            .agg(
                mean_value=(value_column, "mean"),
                std_value=(value_column, "std"),
                n_runs=(value_column, "size"),
            )
        )

        summary["ci95"] = (
            1.96
            * summary["std_value"].fillna(0.0)
            / np.sqrt(summary["n_runs"])
        )

        return summary.rename(
            columns={
                "mean_value": f"mean_{value_column}",
                "std_value": f"std_{value_column}",
                "ci95": f"ci95_{value_column}",
            }
        )

    def _paired_deltas_and_tests(
        self,
        family_replicates,
        value_column,
        condition_columns,
        higher_is_better,
    ):
        pair_columns = condition_columns + ["seed", "_hidden_key"]

        pivot = (
            family_replicates
            .pivot_table(
                index=pair_columns,
                columns="family",
                values=value_column,
                aggfunc="mean",
            )
            .reset_index()
        )

        comparisons = [
            ("attention", "attention_minus_convolution"),
            ("recurrent", "recurrent_minus_convolution"),
        ]

        gap_frames = []

        for target_family, comparison in comparisons:
            if (
                "convolutional" not in pivot.columns
                or target_family not in pivot.columns
            ):
                continue

            paired = pivot[
                pair_columns + [
                    "convolutional",
                    target_family,
                ]
            ].dropna().copy()

            if paired.empty:
                continue

            if higher_is_better:
                paired["delta"] = (
                    paired[target_family] - paired["convolutional"]
                )
            else:
                paired["delta"] = (
                    paired["convolutional"] - paired[target_family]
                )

            paired["comparison"] = comparison
            paired["target_family"] = target_family
            paired["reference_family"] = "convolutional"

            gap_frames.append(paired)

        if not gap_frames:
            empty = pd.DataFrame()
            return empty, empty, empty

        gap_runs = pd.concat(gap_frames, ignore_index=True)
        group_columns = condition_columns + ["comparison"]

        gap_summary = (
            gap_runs
            .groupby(group_columns, as_index=False)
            .agg(
                mean_delta=("delta", "mean"),
                std_delta=("delta", "std"),
                median_delta=("delta", "median"),
                n_paired_runs=("delta", "size"),
                win_rate=("delta", lambda values: np.mean(values > 0)),
            )
        )

        gap_summary["ci95_delta"] = (
            1.96
            * gap_summary["std_delta"].fillna(0.0)
            / np.sqrt(gap_summary["n_paired_runs"])
        )

        test_records = []

        for group_key, group in gap_runs.groupby(
            group_columns,
            sort=False,
        ):
            deltas = group["delta"].to_numpy(dtype=float)
            record = dict(zip(group_columns, group_key))

            record["n_paired_runs"] = len(deltas)
            record["mean_delta"] = float(np.mean(deltas))
            record["median_delta"] = float(np.median(deltas))
            record["win_rate"] = float(np.mean(deltas > 0))

            if len(deltas) < 3:
                record["wilcoxon_statistic"] = np.nan
                record["p_value"] = np.nan
            elif np.allclose(deltas, 0.0):
                record["wilcoxon_statistic"] = 0.0
                record["p_value"] = 1.0
            else:
                try:
                    statistic, p_value = wilcoxon(
                        deltas,
                        alternative="greater",
                        zero_method="wilcox",
                    )
                    record["wilcoxon_statistic"] = statistic
                    record["p_value"] = p_value
                except ValueError:
                    record["wilcoxon_statistic"] = np.nan
                    record["p_value"] = np.nan

            test_records.append(record)

        tests = pd.DataFrame(test_records)

        if not tests.empty:
            tests["p_value_bh"] = self._bh_adjust(
                tests["p_value"].to_numpy(dtype=float)
            )

            tests["significant_bh_0_05"] = (
                tests["p_value_bh"] < 0.05
            )

        return gap_runs, gap_summary, tests

    def analyze_context_horizon(
        self,
        regime_name="Long-Range",
    ):
        base = self._ensure_context_horizon_data()

        rmse_conditions = [
            "dataset",
            "lags",
            "horizon",
        ]

        (
            rmse_architecture_replicates,
            rmse_family_replicates,
        ) = self._family_replicates(
            frame=base,
            value_column="test_rmse",
            condition_columns=rmse_conditions,
        )

        rmse_family_replicates["best_family_rmse"] = (
            rmse_family_replicates
            .groupby(
                rmse_conditions
                + [
                    "seed",
                    "_hidden_key",
                ]
            )["test_rmse"]
            .transform("min")
        )

        rmse_family_replicates["normalized_rmse"] = (
            rmse_family_replicates["test_rmse"]
            / rmse_family_replicates["best_family_rmse"]
        )

        rmse_family_summary = self._family_summary(
            family_replicates=rmse_family_replicates,
            value_column="test_rmse",
            condition_columns=rmse_conditions,
        )

        normalized_rmse_summary = self._family_summary(
            family_replicates=rmse_family_replicates,
            value_column="normalized_rmse",
            condition_columns=rmse_conditions,
        )

        (
            rmse_gap_runs,
            rmse_gap_summary,
            rmse_tests,
        ) = self._paired_deltas_and_tests(
            family_replicates=rmse_family_replicates,
            value_column="test_rmse",
            condition_columns=rmse_conditions,
            higher_is_better=False,
        )

        if "test_r2_per_horizon" not in base.columns:
            raise ValueError(
                "The column 'test_r2_per_horizon' was not loaded."
            )

        r2_base = base[
            base["test_r2_per_horizon"].apply(
                lambda values: (
                    isinstance(
                        values,
                        (list, tuple, np.ndarray),
                    )
                    and len(values) > 0
                )
            )
        ].copy()

        if r2_base.empty:
            raise ValueError(
                "No valid test_r2_per_horizon values were found."
            )

        r2_base = r2_base.reset_index(drop=True)
        r2_base["_run_id"] = np.arange(len(r2_base))

        r2_long = r2_base.explode("test_r2_per_horizon").copy()

        r2_long["horizon_step"] = (
            r2_long
            .groupby("_run_id")
            .cumcount()
            .add(1)
        )

        r2_long["test_r2"] = pd.to_numeric(
            r2_long["test_r2_per_horizon"],
            errors="coerce",
        )

        r2_long = r2_long.dropna(subset=["test_r2"]).copy()

        r2_long = r2_long[
            r2_long["horizon_step"] <= r2_long["horizon"]
        ].copy()

        r2_long["horizon_step"] = (
            r2_long["horizon_step"].astype(int)
        )

        r2_conditions = [
            "dataset",
            "lags",
            "horizon",
            "horizon_step",
        ]

        (
            r2_architecture_replicates,
            r2_family_replicates,
        ) = self._family_replicates(
            frame=r2_long,
            value_column="test_r2",
            condition_columns=r2_conditions,
        )

        r2_family_summary = self._family_summary(
            family_replicates=r2_family_replicates,
            value_column="test_r2",
            condition_columns=r2_conditions,
        )

        (
            r2_gap_runs,
            r2_gap_summary,
            r2_tests,
        ) = self._paired_deltas_and_tests(
            family_replicates=r2_family_replicates,
            value_column="test_r2",
            condition_columns=r2_conditions,
            higher_is_better=True,
        )

        outputs = {
            "rmse_architecture_replicates": (
                rmse_architecture_replicates
            ),
            "rmse_family_replicates": rmse_family_replicates,
            "rmse_family_summary": rmse_family_summary,
            "normalized_rmse_summary": normalized_rmse_summary,
            "rmse_gap_runs": rmse_gap_runs,
            "rmse_gap_summary": rmse_gap_summary,
            "rmse_tests": rmse_tests,
            "r2_long": r2_long,
            "r2_architecture_replicates": r2_architecture_replicates,
            "r2_family_replicates": r2_family_replicates,
            "r2_family_summary": r2_family_summary,
            "r2_gap_runs": r2_gap_runs,
            "r2_gap_summary": r2_gap_summary,
            "r2_tests": r2_tests,
        }

        for name, table in outputs.items():
            if isinstance(table, pd.DataFrame):
                outputs[name] = table.assign(regime=regime_name)

        return outputs

    def export_context_horizon_analysis(
        self,
        analysis,
        output_dir="stats_context_horizon_long_range",
        output_prefix="long_range",
    ):
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        for name, table in analysis.items():
            if not isinstance(table, pd.DataFrame):
                continue

            output_path = os.path.join(
                output_dir,
                f"{output_prefix}_{name}.csv",
            )

            table.to_csv(output_path, index=False)
            saved_files[name] = output_path

        return saved_files

    def plot_context_horizon_analysis(
        self,
        analysis,
        output_dir="stats_context_horizon_long_range",
        output_prefix="long_range",
    ):
        os.makedirs(output_dir, exist_ok=True)

        family_order = [
            "convolutional",
            "recurrent",
            "attention",
        ]

        family_labels = {
            "convolutional": "Convolutional",
            "recurrent": "Recurrent",
            "attention": "Attention",
        }

        family_colors = {
            "convolutional": "#CA0324",
            "recurrent": "#525252",
            "attention": "#D4AF37",
        }

        comparison_labels = {
            "attention_minus_convolution": (
                "Attention − Convolution"
            ),
            "recurrent_minus_convolution": (
                "Recurrent − Convolution"
            ),
        }

        comparison_colors = {
            "attention_minus_convolution": "#D4AF37",
            "recurrent_minus_convolution": "#525252",
        }

        saved_figures = []

        rmse_summary = analysis["rmse_family_summary"]

        for dataset in sorted(rmse_summary["dataset"].unique()):
            dataset_data = rmse_summary[
                rmse_summary["dataset"] == dataset
            ].copy()

            horizons = sorted(dataset_data["horizon"].unique())

            if not horizons:
                continue

            columns = min(3, len(horizons))
            rows = int(np.ceil(len(horizons) / columns))

            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(6.0 * columns, 4.4 * rows),
                squeeze=False,
            )

            axes = axes.ravel()

            for axis, horizon in zip(axes, horizons):
                panel = dataset_data[
                    dataset_data["horizon"] == horizon
                ]

                for family in family_order:
                    line = panel[
                        panel["family"] == family
                    ].sort_values("lags")

                    if line.empty:
                        continue

                    axis.errorbar(
                        line["lags"],
                        line["mean_test_rmse"],
                        yerr=line["ci95_test_rmse"],
                        marker="o",
                        linewidth=2.0,
                        capsize=4,
                        label=family_labels[family],
                        color=family_colors[family],
                    )

                axis.set_title(f"H = {int(horizon)}")
                axis.set_xlabel("Context length L")
                axis.set_ylabel("Mean test RMSE")
                axis.grid(alpha=0.25)

            for axis in axes[len(horizons):]:
                axis.remove()

            handles, labels = axes[0].get_legend_handles_labels()

            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=3,
                    frameon=False,
                )

            figure.suptitle(
                f"{dataset}: RMSE by context length and horizon",
                y=0.99,
                fontsize=15,
            )

            plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

            output_path = os.path.join(
                output_dir,
                f"{output_prefix}_{dataset}_rmse_by_context_horizon.pdf",
            )

            figure.savefig(
                output_path,
                dpi=220,
                bbox_inches="tight",
            )

            plt.close(figure)
            saved_figures.append(output_path)

        r2_summary = analysis["r2_family_summary"]

        for dataset in sorted(r2_summary["dataset"].unique()):
            dataset_data = r2_summary[
                r2_summary["dataset"] == dataset
            ].copy()

            for horizon in sorted(dataset_data["horizon"].unique()):
                horizon_data = dataset_data[
                    dataset_data["horizon"] == horizon
                ].copy()

                steps = sorted(horizon_data["horizon_step"].unique())

                if not steps:
                    continue

                columns = min(4, len(steps))
                rows = int(np.ceil(len(steps) / columns))

                figure, axes = plt.subplots(
                    rows,
                    columns,
                    figsize=(5.4 * columns, 4.2 * rows),
                    squeeze=False,
                )

                axes = axes.ravel()

                for axis, step in zip(axes, steps):
                    panel = horizon_data[
                        horizon_data["horizon_step"] == step
                    ]

                    for family in family_order:
                        line = panel[
                            panel["family"] == family
                        ].sort_values("lags")

                        if line.empty:
                            continue

                        axis.errorbar(
                            line["lags"],
                            line["mean_test_r2"],
                            yerr=line["ci95_test_r2"],
                            marker="o",
                            linewidth=2.0,
                            capsize=4,
                            label=family_labels[family],
                            color=family_colors[family],
                        )

                    axis.axhline(
                        0.0,
                        color="black",
                        linewidth=1.0,
                        linestyle="--",
                    )

                    axis.set_title(f"h = {int(step)}")
                    axis.set_xlabel("Context length L")
                    axis.set_ylabel("Mean $R^2_h$")
                    axis.grid(alpha=0.25)

                for axis in axes[len(steps):]:
                    axis.remove()

                handles, labels = axes[0].get_legend_handles_labels()

                if handles:
                    figure.legend(
                        handles,
                        labels,
                        loc="upper center",
                        ncol=3,
                        frameon=False,
                    )

                figure.suptitle(
                    f"{dataset}: horizon-specific R² for H = {int(horizon)}",
                    y=0.99,
                    fontsize=15,
                )

                plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

                output_path = os.path.join(
                    output_dir,
                    (
                        f"{output_prefix}_{dataset}_"
                        f"r2_by_context_H{int(horizon)}.pdf"
                    ),
                )

                figure.savefig(
                    output_path,
                    dpi=220,
                    bbox_inches="tight",
                )

                plt.close(figure)
                saved_figures.append(output_path)

        r2_gap_summary = analysis["r2_gap_summary"]

        if not r2_gap_summary.empty:
            for dataset in sorted(
                r2_gap_summary["dataset"].unique()
            ):
                dataset_data = r2_gap_summary[
                    r2_gap_summary["dataset"] == dataset
                ].copy()

                for horizon in sorted(
                    dataset_data["horizon"].unique()
                ):
                    horizon_data = dataset_data[
                        dataset_data["horizon"] == horizon
                    ].copy()

                    steps = sorted(
                        horizon_data["horizon_step"].unique()
                    )

                    if not steps:
                        continue

                    columns = min(4, len(steps))
                    rows = int(np.ceil(len(steps) / columns))

                    figure, axes = plt.subplots(
                        rows,
                        columns,
                        figsize=(5.4 * columns, 4.2 * rows),
                        squeeze=False,
                    )

                    axes = axes.ravel()

                    for axis, step in zip(axes, steps):
                        panel = horizon_data[
                            horizon_data["horizon_step"] == step
                        ]

                        for comparison in [
                            "attention_minus_convolution",
                            "recurrent_minus_convolution",
                        ]:
                            line = panel[
                                panel["comparison"] == comparison
                            ].sort_values("lags")

                            if line.empty:
                                continue

                            axis.errorbar(
                                line["lags"],
                                line["mean_delta"],
                                yerr=line["ci95_delta"],
                                marker="o",
                                linewidth=2.0,
                                capsize=4,
                                label=comparison_labels[comparison],
                                color=comparison_colors[comparison],
                            )

                        axis.axhline(
                            0.0,
                            color="black",
                            linewidth=1.0,
                            linestyle="--",
                        )

                        axis.set_title(f"h = {int(step)}")
                        axis.set_xlabel("Context length L")
                        axis.set_ylabel("Family advantage in $R^2_h$")
                        axis.grid(alpha=0.25)

                    for axis in axes[len(steps):]:
                        axis.remove()

                    handles, labels = axes[0].get_legend_handles_labels()

                    if handles:
                        figure.legend(
                            handles,
                            labels,
                            loc="upper center",
                            ncol=2,
                            frameon=False,
                        )

                    figure.suptitle(
                        (
                            f"{dataset}: advantage over convolution "
                            f"for H = {int(horizon)}"
                        ),
                        y=0.99,
                        fontsize=15,
                    )

                    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

                    output_path = os.path.join(
                        output_dir,
                        (
                            f"{output_prefix}_{dataset}_"
                            f"r2_advantage_H{int(horizon)}.pdf"
                        ),
                    )

                    figure.savefig(
                        output_path,
                        dpi=220,
                        bbox_inches="tight",
                    )

                    plt.close(figure)
                    saved_figures.append(output_path)

        return saved_figures

    def run_context_horizon_analysis(
        self,
        regime_name="Long-Range",
        output_dir="stats_context_horizon_long_range",
        output_prefix="long_range",
    ):
        analysis = self.analyze_context_horizon(
            regime_name=regime_name
        )

        csv_files = self.export_context_horizon_analysis(
            analysis=analysis,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )

        figure_files = self.plot_context_horizon_analysis(
            analysis=analysis,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )

        print("\n[CONTEXT-HORIZON ANALYSIS]")
        print(f"Regime: {regime_name}")
        print(f"Output directory: {output_dir}")
        print(f"CSV files: {len(csv_files)}")
        print(f"Figures: {len(figure_files)}")

        return analysis, csv_files, figure_files

    def run_long_range_context_horizon(
        self,
        output_dir="stats_context_horizon_long_range",
        output_prefix="long_range",
    ):
        return self.run_context_horizon_analysis(
            regime_name="Long-Range",
            output_dir=output_dir,
            output_prefix=output_prefix,
        )


if __name__ == "__main__":

    pd.set_option("display.max_rows", 10000)
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", lambda value: f"{value:.6f}")

    roots_long_range = [
        "results_chickenpox_long",
        "results_wikimaths_long",
        "results_englandcovid_long",
        "results_montevideobus_long",
    ]

    output_directories = [
        "stats_cd",
        "stats_tables",
        "stats_spectral_alignment",
        "stats_context_horizon_long_range",
        "stats_input_projection",
    ]

    for directory in output_directories:
        os.makedirs(directory, exist_ok=True)

    analyzer_long_range = ResultsAnalyzer(roots_long_range)
    dataframe = analyzer_long_range.load()

    if dataframe is None or dataframe.empty:
        raise ValueError(
            "No Long-Range results were loaded. "
            "Check the paths in roots_long_range."
        )

    print("\n" + "=" * 120)
    print("LONG-RANGE RESULTS ANALYZER")
    print("=" * 120)

    print("\n[AVAILABLE DATASETS]")
    print(sorted(dataframe["dataset"].unique()))

    print("\n[AVAILABLE ARCHITECTURES]")
    print(sorted(dataframe["architecture"].unique()))

    print("\n[AVAILABLE FAMILIES]")
    print(sorted(dataframe["family"].dropna().unique()))

    print("\n[AVAILABLE CONTEXT LENGTHS]")
    print(sorted(dataframe["lags"].dropna().unique()))

    print("\n[AVAILABLE FORECASTING HORIZONS]")
    print(sorted(dataframe["horizon"].dropna().unique()))

    print("\n[NUMBER OF RUNS PER DATASET]")
    print(
        dataframe
        .groupby("dataset")
        .size()
        .sort_values(ascending=False)
    )

    print("\n[NUMBER OF RUNS PER DATASET, ARCHITECTURE, CONTEXT, AND HORIZON]")
    print(
        dataframe
        .groupby(
            [
                "dataset",
                "architecture",
                "family",
                "lags",
                "horizon",
            ]
        )
        .size()
        .rename("n_runs")
        .reset_index()
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "family",
                "architecture",
            ]
        )
        .to_string(index=False)
    )

    architecture_summary = (
        dataframe
        .groupby(
            [
                "dataset",
                "architecture",
                "family",
                "lags",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", "std"),
            mean_r2_global=("test_r2_global", "mean"),
            std_r2_global=("test_r2_global", "std"),
            mean_r2_horizon=("test_r2_mean_horizon", "mean"),
            std_r2_horizon=("test_r2_mean_horizon", "std"),
            n_runs=("test_rmse", "size"),
        )
    )

    architecture_summary["ci95_rmse"] = (
        1.96
        * architecture_summary["std_rmse"].fillna(0.0)
        / np.sqrt(architecture_summary["n_runs"])
    )

    architecture_summary["ci95_r2_global"] = (
        1.96
        * architecture_summary["std_r2_global"].fillna(0.0)
        / np.sqrt(architecture_summary["n_runs"])
    )

    architecture_summary = architecture_summary.sort_values(
        [
            "dataset",
            "lags",
            "horizon",
            "mean_rmse",
        ]
    )

    print("\n" + "=" * 120)
    print("[ARCHITECTURE SUMMARY BY DATASET, CONTEXT, AND HORIZON]")
    print("=" * 120)
    print(architecture_summary.to_string(index=False))

    family_global_summary = (
        dataframe
        .groupby(
            [
                "dataset",
                "family",
                "lags",
                "horizon",
            ],
            as_index=False,
        )
        .agg(
            mean_rmse=("test_rmse", "mean"),
            std_rmse=("test_rmse", "std"),
            mean_r2_global=("test_r2_global", "mean"),
            std_r2_global=("test_r2_global", "std"),
            mean_r2_horizon=("test_r2_mean_horizon", "mean"),
            std_r2_horizon=("test_r2_mean_horizon", "std"),
            n_runs=("test_rmse", "size"),
        )
    )

    family_global_summary["ci95_rmse"] = (
        1.96
        * family_global_summary["std_rmse"].fillna(0.0)
        / np.sqrt(family_global_summary["n_runs"])
    )

    family_global_summary["ci95_r2_global"] = (
        1.96
        * family_global_summary["std_r2_global"].fillna(0.0)
        / np.sqrt(family_global_summary["n_runs"])
    )

    family_global_summary = family_global_summary.sort_values(
        [
            "dataset",
            "lags",
            "horizon",
            "mean_rmse",
        ]
    )

    print("\n" + "=" * 120)
    print("[GLOBAL FAMILY SUMMARY BY DATASET, CONTEXT, AND HORIZON]")
    print("=" * 120)
    print(family_global_summary.to_string(index=False))

    analyzer_long_range.plot_cd_subplots(
        output=(
            "stats_cd/"
            "long_range_cd_diagrams_all_datasets.pdf"
        )
    )

    analyzer_long_range.export_rmse_table(
        output_csv=(
            "stats_tables/"
            "long_range_rmse_summary.csv"
        )
    )

    analyzer_long_range.topk_rmse(k=3)

    spectral_results = (
        analyzer_long_range.spectral_performance_alignment(
            regime_name="Long-Range",
            output_dir="stats_spectral_alignment",
            output_prefix="long_range",
        )
    )

    print("\n" + "=" * 120)
    print("[SPECTRAL ALIGNMENT: NORMALIZED RMSE]")
    print("=" * 120)
    print(
        spectral_results["normalized_rmse"]
        .sort_values(["dataset", "rank"])
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[SPECTRAL ALIGNMENT: FAMILY MEAN RANK]")
    print("=" * 120)
    print(
        spectral_results["family_rank_mean"]
        .sort_values(["dataset", "family_mean_rank"])
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[SPECTRAL ALIGNMENT: PERFORMANCE DISPERSION]")
    print("=" * 120)
    print(
        spectral_results["performance_dispersion"]
        .sort_values("dataset")
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[SPECTRAL-ARCHITECTURAL ALIGNMENT]")
    print("=" * 120)
    print(
        spectral_results["spectral_architectural_alignment"]
        .sort_values("dataset")
        .to_string(index=False)
    )

    (
        long_range_analysis,
        long_range_csvs,
        long_range_figures,
    ) = analyzer_long_range.run_long_range_context_horizon(
        output_dir="stats_context_horizon_long_range",
        output_prefix="long_range",
    )

    print("\n" + "=" * 120)
    print("[FAMILY RMSE BY DATASET, CONTEXT, AND HORIZON]")
    print("=" * 120)
    print(
        long_range_analysis["rmse_family_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "mean_test_rmse",
            ]
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[NORMALIZED FAMILY RMSE BY DATASET, CONTEXT, AND HORIZON]")
    print("=" * 120)
    print(
        long_range_analysis["normalized_rmse_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "mean_normalized_rmse",
            ]
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[RMSE ADVANTAGE OVER CONVOLUTIONAL MODELS]")
    print("=" * 120)
    print(
        long_range_analysis["rmse_gap_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "comparison",
            ]
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[PAIRED WILCOXON TESTS: RMSE ADVANTAGE OVER CONVOLUTIONAL MODELS]")
    print("=" * 120)
    print(
        long_range_analysis["rmse_tests"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "comparison",
            ]
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[HORIZON-SPECIFIC FAMILY R² BY DATASET, CONTEXT, HORIZON, AND STEP]")
    print("=" * 120)
    print(
        long_range_analysis["r2_family_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "horizon_step",
                "mean_test_r2",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                False,
            ],
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[HORIZON-SPECIFIC R² ADVANTAGE OVER CONVOLUTIONAL MODELS]")
    print("=" * 120)
    print(
        long_range_analysis["r2_gap_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "horizon_step",
                "comparison",
            ]
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[PAIRED WILCOXON TESTS: HORIZON-SPECIFIC R² ADVANTAGE]")
    print("=" * 120)
    print(
        long_range_analysis["r2_tests"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "horizon_step",
                "comparison",
            ]
        )
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("[BEST FAMILY BY MEAN RMSE]")
    print("=" * 120)

    best_family_rmse = (
        long_range_analysis["rmse_family_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "mean_test_rmse",
            ]
        )
        .groupby(
            [
                "dataset",
                "lags",
                "horizon",
            ],
            as_index=False,
        )
        .first()
    )

    print(best_family_rmse.to_string(index=False))

    print("\n" + "=" * 120)
    print("[BEST FAMILY BY HORIZON-SPECIFIC R²]")
    print("=" * 120)

    best_family_r2 = (
        long_range_analysis["r2_family_summary"]
        .sort_values(
            [
                "dataset",
                "lags",
                "horizon",
                "horizon_step",
                "mean_test_r2",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                False,
            ],
        )
        .groupby(
            [
                "dataset",
                "lags",
                "horizon",
                "horizon_step",
            ],
            as_index=False,
        )
        .first()
    )

    print(best_family_r2.to_string(index=False))

    print("\n" + "=" * 120)
    print("[GENERATED CSV FILES]")
    print("=" * 120)

    for name, path in long_range_csvs.items():
        print(f"{name}: {path}")

    print("\n" + "=" * 120)
    print("[GENERATED FIGURES]")
    print("=" * 120)

    for path in long_range_figures:
        print(path)

    print("\n" + "=" * 120)
    print("LONG-RANGE ANALYSIS FINISHED")
    print("=" * 120)