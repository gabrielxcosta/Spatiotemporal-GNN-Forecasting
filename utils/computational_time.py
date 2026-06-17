import os
import json
import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    if arch in RECURRENT_MODELS:
        return "recurrent"

    return "convolutional"


def parse_root_name(root):
    name = os.path.basename(os.path.normpath(root))

    if not name.startswith("results_"):
        return name, "unknown"

    name = name.replace("results_", "")

    if name.endswith("_long"):
        dataset = name.replace("_long", "")
        regime = "long"
    else:
        dataset = name
        regime = "short_mid"

    return dataset, regime


def _process_metrics_file(args):
    dataset, regime, arch, config, seed_path = args

    metrics_path = os.path.join(seed_path, "metrics.json")

    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    runtime_sec = data.get("runtime_sec", None)
    epochs_ran = data.get("epochs_ran", None)

    if runtime_sec is None:
        return None

    if epochs_ran is None or epochs_ran <= 0:
        time_per_epoch = np.nan
    else:
        time_per_epoch = runtime_sec / epochs_ran

    hidden_match = re.search(r"hid(\d+)", config)
    hidden = int(hidden_match.group(1)) if hidden_match else None

    row = {
        "dataset": dataset,
        "regime": regime,
        "architecture": arch,
        "family": get_family(arch),
        "config_name": config,
        "hidden": hidden,
        "seed": data.get("seed", None),
        "runtime_sec": runtime_sec,
        "epochs_ran": epochs_ran,
        "time_per_epoch": time_per_epoch,
    }

    if "config" in data:
        for k, v in data["config"].items():
            row[k] = v

    return row


class ComputationalTimeAnalyzer:

    def __init__(self, roots, num_workers=None):
        if isinstance(roots, str):
            roots = [roots]

        self.roots = roots
        self.df = None
        self.num_workers = num_workers or os.cpu_count() or 1

    def load(self):
        tasks = []

        for root in self.roots:
            dataset, regime = parse_root_name(root)

            if not os.path.isdir(root):
                print(f"[SKIP] Directory not found: {root}")
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
                            (
                                dataset,
                                regime,
                                arch,
                                config,
                                seed_path,
                            )
                        )

        rows = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
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

        if not self.df.empty:
            print("\n[Regimes]")
            print(self.df["regime"].value_counts())

            print("\n[Datasets]")
            print(self.df["dataset"].value_counts())

            print("\n[Architectures]")
            print(self.df["architecture"].nunique())

            print("\n[Columns]")
            print(self.df.columns.tolist())

        return self.df

    def _add_weighted_time_per_epoch(self, summary):
        summary["weighted_time_per_epoch"] = np.where(
            summary["total_epochs"] > 0,
            summary["total_runtime"] / summary["total_epochs"],
            np.nan,
        )

        return summary

    def summarize_by_model_config(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "regime",
                    "dataset",
                    "architecture",
                    "family",
                    "config_name",
                    "hidden",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            [
                "regime",
                "dataset",
                "architecture",
                "time_per_epoch_mean",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )

    def summarize_by_model_dataset_regime(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "regime",
                    "dataset",
                    "architecture",
                    "family",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            [
                "regime",
                "dataset",
                "weighted_time_per_epoch",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )

    def summarize_by_model_regime(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "regime",
                    "architecture",
                    "family",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            [
                "regime",
                "weighted_time_per_epoch",
            ],
            ascending=[
                True,
                True,
            ],
        )

    def summarize_by_model_overall(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "architecture",
                    "family",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            "weighted_time_per_epoch",
            ascending=True,
        )

    def summarize_by_family_dataset_regime(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "regime",
                    "dataset",
                    "family",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            [
                "regime",
                "dataset",
                "weighted_time_per_epoch",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )

    def summarize_by_family_regime(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "regime",
                    "family",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            [
                "regime",
                "weighted_time_per_epoch",
            ],
            ascending=[
                True,
                True,
            ],
        )

    def summarize_by_family_overall(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = (
            self.df
            .groupby(
                [
                    "family",
                ],
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                runtime_mean=("runtime_sec", "mean"),
                runtime_std=("runtime_sec", "std"),
                epochs_mean=("epochs_ran", "mean"),
                epochs_std=("epochs_ran", "std"),
                time_per_epoch_mean=("time_per_epoch", "mean"),
                time_per_epoch_std=("time_per_epoch", "std"),
                total_runtime=("runtime_sec", "sum"),
                total_epochs=("epochs_ran", "sum"),
            )
            .reset_index()
        )

        summary = self._add_weighted_time_per_epoch(summary)

        return summary.sort_values(
            "weighted_time_per_epoch",
            ascending=True,
        )

    def export_tables(self, output_dir="computational_time"):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        os.makedirs(output_dir, exist_ok=True)

        seed_level = self.df.copy()

        model_config = self.summarize_by_model_config()
        model_dataset_regime = self.summarize_by_model_dataset_regime()
        model_regime = self.summarize_by_model_regime()
        model_overall = self.summarize_by_model_overall()

        family_dataset_regime = self.summarize_by_family_dataset_regime()
        family_regime = self.summarize_by_family_regime()
        family_overall = self.summarize_by_family_overall()

        seed_level.to_csv(
            os.path.join(
                output_dir,
                "computational_time_seed_level.csv",
            ),
            index=False,
        )

        model_config.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_model_config.csv",
            ),
            index=False,
        )

        model_dataset_regime.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_model_dataset_regime.csv",
            ),
            index=False,
        )

        model_regime.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_model_regime.csv",
            ),
            index=False,
        )

        model_overall.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_model_overall.csv",
            ),
            index=False,
        )

        family_dataset_regime.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_family_dataset_regime.csv",
            ),
            index=False,
        )

        family_regime.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_family_regime.csv",
            ),
            index=False,
        )

        family_overall.to_csv(
            os.path.join(
                output_dir,
                "computational_time_by_family_overall.csv",
            ),
            index=False,
        )

        model_regime_pivot = model_regime.pivot_table(
            index="architecture",
            columns="regime",
            values="weighted_time_per_epoch",
        )

        model_regime_pivot.to_csv(
            os.path.join(
                output_dir,
                "computational_time_model_regime_pivot.csv",
            )
        )

        model_overall_pivot = model_overall.pivot_table(
            index="architecture",
            values="weighted_time_per_epoch",
        )

        model_overall_pivot.to_csv(
            os.path.join(
                output_dir,
                "computational_time_model_overall_pivot.csv",
            )
        )

        print(f"\nSaved tables to: {output_dir}")

    def print_model_regime_summary(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = self.summarize_by_model_regime()

        columns = [
            "regime",
            "architecture",
            "family",
            "runs",
            "time_per_epoch_mean",
            "time_per_epoch_std",
            "weighted_time_per_epoch",
            "runtime_mean",
            "epochs_mean",
        ]

        print("\n[Computational time by model and regime]")
        print(summary[columns].to_string(index=False))

    def print_model_overall_summary(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = self.summarize_by_model_overall()

        columns = [
            "architecture",
            "family",
            "runs",
            "time_per_epoch_mean",
            "time_per_epoch_std",
            "weighted_time_per_epoch",
            "runtime_mean",
            "epochs_mean",
        ]

        print("\n[Computational time by model overall]")
        print(summary[columns].to_string(index=False))

    def print_family_summaries(self):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        by_regime = self.summarize_by_family_regime()
        overall = self.summarize_by_family_overall()

        columns_regime = [
            "regime",
            "family",
            "runs",
            "time_per_epoch_mean",
            "time_per_epoch_std",
            "weighted_time_per_epoch",
            "runtime_mean",
            "epochs_mean",
        ]

        columns_overall = [
            "family",
            "runs",
            "time_per_epoch_mean",
            "time_per_epoch_std",
            "weighted_time_per_epoch",
            "runtime_mean",
            "epochs_mean",
        ]

        print("\n[Computational time by family and regime]")
        print(by_regime[columns_regime].to_string(index=False))

        print("\n[Computational time by family overall]")
        print(overall[columns_overall].to_string(index=False))

    def print_topk_fastest_models(self, k=5):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        summary = self.summarize_by_model_dataset_regime()

        print(f"\n[Top {k} fastest architectures by regime and dataset]")

        for regime in sorted(summary["regime"].unique()):
            for dataset in sorted(summary["dataset"].unique()):
                group = summary[
                    (summary["regime"] == regime)
                    & (summary["dataset"] == dataset)
                ]

                if group.empty:
                    continue

                ranking = (
                    group
                    .sort_values(
                        "weighted_time_per_epoch",
                        ascending=True,
                    )
                    .head(k)
                    [
                        [
                            "architecture",
                            "family",
                            "time_per_epoch_mean",
                            "time_per_epoch_std",
                            "weighted_time_per_epoch",
                            "runs",
                        ]
                    ]
                )

                print(f"\nRegime: {regime} | Dataset: {dataset}")
                print(ranking.to_string(index=False))

    def plot_by_model_regime(self, output_dir="computational_time/plots"):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        os.makedirs(output_dir, exist_ok=True)

        summary = self.summarize_by_model_regime()

        regime_order = [
            "short_mid",
            "long",
        ]

        family_colors = {
            "attention": "#D4AF37",
            "recurrent": "#525252",
            "convolutional": "#CA0324",
        }

        for regime in regime_order:
            group = summary[summary["regime"] == regime].copy()

            if group.empty:
                continue

            group = group.sort_values(
                "weighted_time_per_epoch",
                ascending=True,
            )

            colors = [
                family_colors[family]
                for family in group["family"]
            ]

            yerr = group["time_per_epoch_std"].fillna(0.0)

            plt.figure(figsize=(14, 6))

            plt.bar(
                group["architecture"],
                group["weighted_time_per_epoch"],
                yerr=yerr,
                capsize=3,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
            )

            plt.ylabel("Weighted time per epoch (s)")
            plt.xlabel("Architecture")
            plt.title(f"Computational time per epoch by model ({regime})")
            plt.xticks(rotation=60, ha="right")
            plt.tight_layout()

            output = os.path.join(
                output_dir,
                f"computational_time_by_model_{regime}.pdf",
            )

            plt.savefig(
                output,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

            print(f"Saved plot → {output}")

    def plot_by_model_overall(self, output_dir="computational_time/plots"):
        if self.df is None or self.df.empty:
            raise ValueError("Run load() first.")

        os.makedirs(output_dir, exist_ok=True)

        summary = self.summarize_by_model_overall()

        family_colors = {
            "attention": "#D4AF37",
            "recurrent": "#525252",
            "convolutional": "#CA0324",
        }

        summary = summary.sort_values(
            "weighted_time_per_epoch",
            ascending=True,
        )

        colors = [
            family_colors[family]
            for family in summary["family"]
        ]

        yerr = summary["time_per_epoch_std"].fillna(0.0)

        plt.figure(figsize=(14, 6))

        plt.bar(
            summary["architecture"],
            summary["weighted_time_per_epoch"],
            yerr=yerr,
            capsize=3,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )

        plt.ylabel("Weighted time per epoch (s)")
        plt.xlabel("Architecture")
        plt.title("Computational time per epoch by model (overall)")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()

        output = os.path.join(
            output_dir,
            "computational_time_by_model_overall.pdf",
        )

        plt.savefig(
            output,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved plot → {output}")