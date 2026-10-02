import os
import json
import re
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import friedmanchisquare, spearmanr, pearsonr, wilcoxon
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
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    hidden_match = re.search(r"hid(\d+)", config)
    hidden = int(hidden_match.group(1)) if hidden_match else None

    r2_per_horizon = data.get("test_r2_per_horizon", [])

    if not isinstance(r2_per_horizon, (list, tuple)):
        r2_per_horizon = []

    row = {
        "dataset": dataset,
        "architecture": arch,
        "family": get_family(arch),
        "config_name": config,
        "hidden": hidden,
        "seed": data.get("seed", None),
        "test_mse": data.get("test_mse", None),
        "test_rmse": data.get("test_rmse", None),
        "test_mae": data.get("test_mae", None),
        "test_mape": data.get("test_mape", None),
        "test_r2_global": data.get("test_r2_global", None),
        "test_r2_mean_horizon": data.get("test_r2_mean_horizon", None),
        "test_r2_per_horizon": r2_per_horizon,
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
        self.roots = roots
        self.df = None
        self.num_workers = num_workers or os.cpu_count()

    def load(self):

        tasks = []

        for root in self.roots:

            root_name = os.path.basename(os.path.normpath(root))

            if root_name.endswith("_long"):
                continue

            dataset = root_name.replace("results_", "")

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

                        tasks.append(
                            (dataset, arch, config, seed_path)
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
            print("No Short-Mid metrics files were loaded.")
            return self.df

        if "test_r2_global" in self.df.columns:

            print("\n[Top 3 architectures per dataset by R²]")

            for dataset in sorted(self.df["dataset"].unique()):

                group = self.df[
                    self.df["dataset"] == dataset
                ]

                best = (
                    group
                    .groupby("architecture")["test_r2_global"]
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

                                                       
            for t in ax.texts:
                for arch in archs:
                    if arch in t.get_text():
                        t.set_color(color_map[arch])
                        break

                              
            for coll in ax.collections:
                offsets = coll.get_offsets()
                if offsets is None or len(offsets) == 0:
                    continue

                xs = np.asarray(offsets)[:, 0]
                cols = [color_map[min(archs, key=lambda a: abs(avg_rank[a] - x))] for x in xs]
                coll.set_facecolor(cols)
                coll.set_edgecolor("black")
                coll.set_linewidth(0.8)

                                       
                                                                       
            for line in ax.lines:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)

                if xdata.size == 0 or ydata.size == 0:
                    continue

                                                                            
                if np.allclose(ydata, ydata[0]):
                    line.set_color("black")
                    line.set_linewidth(2.2)
                    continue

                                                           
                                                                                               
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

                                                                 
        grouped = (
            self.df
            .groupby(["dataset", "architecture"])["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )

                                   
        grouped["rmse"] = grouped.apply(
            lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
        )

                                                         
        table = grouped.pivot(
            index="architecture",
            columns="dataset",
            values="rmse"
        )

                                                          
        desired_order = ["chickenpox", "wikimaths", "englandcovid", "montevideobus"]
        cols = [c for c in desired_order if c in table.columns]
        table = table[cols]

                   
        table.to_csv(output_csv)

        print(f"Saved RMSE table → {output_csv}")

    def topk_rmse(self, k=3):

        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        print(f"\n[Top {k} architectures per dataset by RMSE]")

        for dataset in sorted(self.df["dataset"].unique()):

            group = self.df[self.df["dataset"] == dataset]

                                           
            ranking = (
                group.groupby("architecture")["test_rmse"]
                .mean()
                .sort_values(ascending=True)                  
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

    def _ensure_context_horizon_data(self):

        if self.df is None or self.df.empty:
            self.load()

        if self.df is None or self.df.empty:
            raise ValueError(
                "DataFrame is empty. No Short-Mid results were loaded."
            )

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

        if not np.any(valid):
            return adjusted

        valid_values = p_values[valid]
        order = np.argsort(valid_values)
        ranked = valid_values[order]

        total = len(ranked)

        corrected = ranked * total / np.arange(
            1,
            total + 1,
        )

        corrected = np.minimum.accumulate(
            corrected[::-1]
        )[::-1]

        corrected = np.clip(
            corrected,
            0.0,
            1.0,
        )

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

        summary = summary.rename(
            columns={
                "mean_value": f"mean_{value_column}",
                "std_value": f"std_{value_column}",
                "ci95": f"ci95_{value_column}",
            }
        )

        return summary

    def _paired_deltas_and_tests(
        self,
        family_replicates,
        value_column,
        condition_columns,
        higher_is_better,
    ):

        pair_columns = condition_columns + [
            "seed",
            "_hidden_key",
        ]

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
            (
                "attention",
                "attention_minus_convolution",
            ),
            (
                "recurrent",
                "recurrent_minus_convolution",
            ),
        ]

        delta_frames = []

        for target_family, comparison in comparisons:

            required_families = {
                "convolutional",
                target_family,
            }

            if not required_families.issubset(
                set(pivot.columns)
            ):
                continue

            paired = pivot[
                pair_columns
                + [
                    "convolutional",
                    target_family,
                ]
            ].dropna().copy()

            if paired.empty:
                continue

            if higher_is_better:
                paired["delta"] = (
                    paired[target_family]
                    - paired["convolutional"]
                )
            else:
                paired["delta"] = (
                    paired["convolutional"]
                    - paired[target_family]
                )

            paired["comparison"] = comparison
            paired["target_family"] = target_family
            paired["reference_family"] = "convolutional"

            delta_frames.append(paired)

        if not delta_frames:
            empty = pd.DataFrame()
            return empty, empty, empty

        delta_runs = pd.concat(
            delta_frames,
            ignore_index=True,
        )

        group_columns = condition_columns + ["comparison"]

        delta_summary = (
            delta_runs
            .groupby(
                group_columns,
                as_index=False,
            )
            .agg(
                mean_delta=("delta", "mean"),
                std_delta=("delta", "std"),
                median_delta=("delta", "median"),
                n_paired_runs=("delta", "size"),
                win_rate=("delta", lambda values: np.mean(values > 0)),
            )
        )

        delta_summary["ci95_delta"] = (
            1.96
            * delta_summary["std_delta"].fillna(0.0)
            / np.sqrt(delta_summary["n_paired_runs"])
        )

        tests = []

        for group_key, group in delta_runs.groupby(
            group_columns,
            sort=False,
        ):

            values = group["delta"].to_numpy(dtype=float)

            record = dict(
                zip(group_columns, group_key)
            )

            record["n_paired_runs"] = len(values)
            record["mean_delta"] = np.mean(values)
            record["median_delta"] = np.median(values)
            record["win_rate"] = np.mean(values > 0)

            if len(values) < 3:
                record["wilcoxon_statistic"] = np.nan
                record["p_value"] = np.nan
            elif np.allclose(values, 0.0):
                record["wilcoxon_statistic"] = 0.0
                record["p_value"] = 1.0
            else:
                try:
                    statistic, p_value = wilcoxon(
                        values,
                        alternative="greater",
                        zero_method="wilcox",
                    )
                    record["wilcoxon_statistic"] = statistic
                    record["p_value"] = p_value
                except ValueError:
                    record["wilcoxon_statistic"] = np.nan
                    record["p_value"] = np.nan

            tests.append(record)

        tests = pd.DataFrame(tests)

        if not tests.empty:
            tests["p_value_bh"] = self._bh_adjust(
                tests["p_value"].to_numpy(dtype=float)
            )

            tests["significant_bh_0_05"] = (
                tests["p_value_bh"] < 0.05
            )

        return delta_runs, delta_summary, tests

    def analyze_context_horizon(
        self,
        regime_name="Short-Mid",
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
                "The test_r2_per_horizon column was not loaded."
            )

        r2_base = base[
            base["test_r2_per_horizon"].apply(
                lambda values: isinstance(
                    values,
                    (
                        list,
                        tuple,
                        np.ndarray,
                    ),
                )
                and len(values) > 0
            )
        ].copy()

        if r2_base.empty:
            raise ValueError(
                "No valid test_r2_per_horizon values were found."
            )

        r2_base = r2_base.reset_index(drop=True)
        r2_base["_run_id"] = np.arange(len(r2_base))

        r2_long = r2_base.explode(
            "test_r2_per_horizon"
        ).copy()

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

        r2_long = r2_long.dropna(
            subset=["test_r2"]
        ).copy()

        r2_long = r2_long[
            r2_long["horizon_step"]
            <= r2_long["horizon"]
        ].copy()

        r2_long["horizon_step"] = (
            r2_long["horizon_step"]
            .astype(int)
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
            "rmse_family_replicates": (
                rmse_family_replicates
            ),
            "rmse_family_summary": rmse_family_summary,
            "normalized_rmse_summary": (
                normalized_rmse_summary
            ),
            "rmse_gap_runs": rmse_gap_runs,
            "rmse_gap_summary": rmse_gap_summary,
            "rmse_tests": rmse_tests,
            "r2_long": r2_long,
            "r2_architecture_replicates": (
                r2_architecture_replicates
            ),
            "r2_family_replicates": (
                r2_family_replicates
            ),
            "r2_family_summary": r2_family_summary,
            "r2_gap_runs": r2_gap_runs,
            "r2_gap_summary": r2_gap_summary,
            "r2_tests": r2_tests,
        }

        for name, table in outputs.items():
            if isinstance(table, pd.DataFrame):
                outputs[name] = table.assign(
                    regime=regime_name
                )

        return outputs

    def export_context_horizon_analysis(
        self,
        analysis,
        output_dir="stats_context_horizon_short_mid",
        output_prefix="short_mid",
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

            table.to_csv(
                output_path,
                index=False,
            )

            saved_files[name] = output_path

        return saved_files

    def plot_context_horizon_analysis(
        self,
        analysis,
        output_dir="stats_context_horizon_short_mid",
        output_prefix="short_mid",
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

        for dataset in sorted(
            rmse_summary["dataset"].unique()
        ):

            dataset_data = rmse_summary[
                rmse_summary["dataset"] == dataset
            ].copy()

            horizons = sorted(
                dataset_data["horizon"].unique()
            )

            if not horizons:
                continue

            ncols = min(3, len(horizons))
            nrows = int(
                np.ceil(len(horizons) / ncols)
            )

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(6.0 * ncols, 4.4 * nrows),
                squeeze=False,
            )

            axes = axes.ravel()

            for ax, horizon in zip(axes, horizons):

                panel = dataset_data[
                    dataset_data["horizon"] == horizon
                ]

                for family in family_order:

                    line = panel[
                        panel["family"] == family
                    ].sort_values("lags")

                    if line.empty:
                        continue

                    ax.errorbar(
                        line["lags"],
                        line["mean_test_rmse"],
                        yerr=line["ci95_test_rmse"],
                        marker="o",
                        linewidth=2.0,
                        capsize=4,
                        label=family_labels[family],
                        color=family_colors[family],
                    )

                ax.set_title(f"H = {int(horizon)}")
                ax.set_xlabel("Context length L")
                ax.set_ylabel("Mean test RMSE")
                ax.grid(alpha=0.25)

            for ax in axes[len(horizons):]:
                ax.remove()

            handles, labels = axes[0].get_legend_handles_labels()

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=3,
                    frameon=False,
                )

            fig.suptitle(
                f"{dataset}: RMSE by context length and horizon",
                y=0.99,
                fontsize=15,
            )

            plt.tight_layout(
                rect=[0.0, 0.0, 1.0, 0.92]
            )

            output_path = os.path.join(
                output_dir,
                (
                    f"{output_prefix}_{dataset}_"
                    "rmse_by_context_horizon.pdf"
                ),
            )

            fig.savefig(
                output_path,
                dpi=220,
                bbox_inches="tight",
            )

            plt.close(fig)

            saved_figures.append(output_path)

        r2_summary = analysis["r2_family_summary"]

        for dataset in sorted(
            r2_summary["dataset"].unique()
        ):

            dataset_data = r2_summary[
                r2_summary["dataset"] == dataset
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

                ncols = min(3, len(steps))
                nrows = int(
                    np.ceil(len(steps) / ncols)
                )

                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(6.0 * ncols, 4.4 * nrows),
                    squeeze=False,
                )

                axes = axes.ravel()

                for ax, step in zip(axes, steps):

                    panel = horizon_data[
                        horizon_data["horizon_step"] == step
                    ]

                    for family in family_order:

                        line = panel[
                            panel["family"] == family
                        ].sort_values("lags")

                        if line.empty:
                            continue

                        ax.errorbar(
                            line["lags"],
                            line["mean_test_r2"],
                            yerr=line["ci95_test_r2"],
                            marker="o",
                            linewidth=2.0,
                            capsize=4,
                            label=family_labels[family],
                            color=family_colors[family],
                        )

                    ax.axhline(
                        0.0,
                        color="black",
                        linewidth=1.0,
                        linestyle="--",
                    )

                    ax.set_title(f"h = {int(step)}")
                    ax.set_xlabel("Context length L")
                    ax.set_ylabel("Mean $R^2_h$")
                    ax.grid(alpha=0.25)

                for ax in axes[len(steps):]:
                    ax.remove()

                handles, labels = axes[0].get_legend_handles_labels()

                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="upper center",
                        ncol=3,
                        frameon=False,
                    )

                fig.suptitle(
                    (
                        f"{dataset}: horizon-specific R² "
                        f"for H = {int(horizon)}"
                    ),
                    y=0.99,
                    fontsize=15,
                )

                plt.tight_layout(
                    rect=[0.0, 0.0, 1.0, 0.92]
                )

                output_path = os.path.join(
                    output_dir,
                    (
                        f"{output_prefix}_{dataset}_"
                        f"r2_by_context_H{int(horizon)}.pdf"
                    ),
                )

                fig.savefig(
                    output_path,
                    dpi=220,
                    bbox_inches="tight",
                )

                plt.close(fig)

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

                    ncols = min(3, len(steps))
                    nrows = int(
                        np.ceil(len(steps) / ncols)
                    )

                    fig, axes = plt.subplots(
                        nrows,
                        ncols,
                        figsize=(6.0 * ncols, 4.4 * nrows),
                        squeeze=False,
                    )

                    axes = axes.ravel()

                    for ax, step in zip(axes, steps):

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

                            ax.errorbar(
                                line["lags"],
                                line["mean_delta"],
                                yerr=line["ci95_delta"],
                                marker="o",
                                linewidth=2.0,
                                capsize=4,
                                label=comparison_labels[
                                    comparison
                                ],
                                color=comparison_colors[
                                    comparison
                                ],
                            )

                        ax.axhline(
                            0.0,
                            color="black",
                            linewidth=1.0,
                            linestyle="--",
                        )

                        ax.set_title(f"h = {int(step)}")
                        ax.set_xlabel("Context length L")
                        ax.set_ylabel(
                            "Family advantage in $R^2_h$"
                        )
                        ax.grid(alpha=0.25)

                    for ax in axes[len(steps):]:
                        ax.remove()

                    handles, labels = (
                        axes[0].get_legend_handles_labels()
                    )

                    if handles:
                        fig.legend(
                            handles,
                            labels,
                            loc="upper center",
                            ncol=2,
                            frameon=False,
                        )

                    fig.suptitle(
                        (
                            f"{dataset}: advantage over convolution "
                            f"for H = {int(horizon)}"
                        ),
                        y=0.99,
                        fontsize=15,
                    )

                    plt.tight_layout(
                        rect=[0.0, 0.0, 1.0, 0.92]
                    )

                    output_path = os.path.join(
                        output_dir,
                        (
                            f"{output_prefix}_{dataset}_"
                            f"r2_advantage_H{int(horizon)}.pdf"
                        ),
                    )

                    fig.savefig(
                        output_path,
                        dpi=220,
                        bbox_inches="tight",
                    )

                    plt.close(fig)

                    saved_figures.append(output_path)

        return saved_figures

    def run_context_horizon_analysis(
        self,
        regime_name="Short-Mid",
        output_dir="stats_context_horizon_short_mid",
        output_prefix="short_mid",
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

    def run_short_mid_context_horizon(
        self,
        output_dir="stats_context_horizon_short_mid",
        output_prefix="short_mid",
    ):

        return self.run_context_horizon_analysis(
            regime_name="Short-Mid",
            output_dir=output_dir,
            output_prefix=output_prefix,
        )


if __name__ == "__main__":

    roots_short_mid = [
        "results_chickenpox",
        "results_wikimaths",
        "results_englandcovid",
        "results_montevideobus",
    ]

    analyzer_short_mid = ResultsAnalyzer(
        roots_short_mid
    )

    analyzer_short_mid.load()

    os.makedirs("stats_tables", exist_ok=True)

    analyzer_short_mid.plot_cd_subplots(
        output=(
            "stats_cd/"
            "short_mid_cd_diagrams_all_datasets.pdf"
        )
    )

    analyzer_short_mid.export_rmse_table(
        output_csv=(
            "stats_tables/"
            "short_mid_rmse_summary.csv"
        )
    )

    analyzer_short_mid.topk_rmse(k=3)

    analyzer_short_mid.spectral_performance_alignment(
        regime_name="Short-Mid",
        output_dir="stats_spectral_alignment",
        output_prefix="short_mid",
    )

    (
        short_mid_analysis,
        short_mid_csvs,
        short_mid_figures,
    ) = analyzer_short_mid.run_short_mid_context_horizon(
        output_dir="stats_context_horizon_short_mid",
        output_prefix="short_mid",
    )