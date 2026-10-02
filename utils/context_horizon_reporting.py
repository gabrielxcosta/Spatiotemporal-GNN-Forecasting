from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHORT_MID_DIR = PROJECT_ROOT / "stats_context_horizon_short_mid"
LONG_RANGE_DIR = PROJECT_ROOT / "stats_context_horizon_long_range"

FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
TABLES_DIR = PROJECT_ROOT / "tables" / "results"


DATASET_ORDER = [
    "chickenpox",
    "wikimaths",
    "englandcovid",
    "montevideobus",
]

DATASET_LABELS = {
    "chickenpox": "Hungary Chickenpox",
    "wikimaths": "Wikipedia Mathematics",
    "englandcovid": "England COVID-19",
    "montevideobus": "Montevideo Bus",
}

COMPARISON_ORDER = [
    "attention_minus_convolution",
    "recurrent_minus_convolution",
]

COMPARISON_LABELS = {
    "attention_minus_convolution": "Attention $-$ Convolution",
    "recurrent_minus_convolution": "Recurrent $-$ Convolution",
}


def read_csv_required(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def normalize_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def validate_columns(frame, required_columns, source_path):
    missing_columns = sorted(
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {source_path}: {missing_columns}"
        )


def load_gap_data(summary_path, tests_path):
    summary = read_csv_required(summary_path)
    tests = read_csv_required(tests_path)

    merge_columns = [
        "dataset",
        "lags",
        "horizon",
        "comparison",
    ]

    validate_columns(
        summary,
        set(
            merge_columns
            + [
                "mean_delta",
                "ci95_delta",
            ]
        ),
        summary_path,
    )

    validate_columns(
        tests,
        set(
            merge_columns
            + [
                "significant_bh_0_05",
            ]
        ),
        tests_path,
    )

    summary = summary.copy()
    tests = tests.copy()

    for frame in [summary, tests]:
        frame["dataset"] = (
            frame["dataset"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        frame["comparison"] = (
            frame["comparison"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        frame["lags"] = pd.to_numeric(
            frame["lags"],
            errors="coerce",
        )

        frame["horizon"] = pd.to_numeric(
            frame["horizon"],
            errors="coerce",
        )

    summary["mean_delta"] = pd.to_numeric(
        summary["mean_delta"],
        errors="coerce",
    )

    summary["ci95_delta"] = pd.to_numeric(
        summary["ci95_delta"],
        errors="coerce",
    )

    tests["significant_bh_0_05"] = normalize_boolean(
        tests["significant_bh_0_05"]
    )

    summary = summary.dropna(
        subset=[
            "dataset",
            "lags",
            "horizon",
            "comparison",
            "mean_delta",
            "ci95_delta",
        ]
    ).copy()

    tests = tests.dropna(
        subset=[
            "dataset",
            "lags",
            "horizon",
            "comparison",
        ]
    ).copy()

    summary["lags"] = summary["lags"].astype(int)
    summary["horizon"] = summary["horizon"].astype(int)

    tests["lags"] = tests["lags"].astype(int)
    tests["horizon"] = tests["horizon"].astype(int)

    duplicate_summary = summary.duplicated(
        merge_columns,
        keep=False,
    )

    if duplicate_summary.any():
        duplicated_rows = summary.loc[
            duplicate_summary,
            merge_columns,
        ]

        raise ValueError(
            "The summary file contains duplicate rows:\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    optional_test_columns = [
        column
        for column in [
            "p_value",
            "p_value_bh",
        ]
        if column in tests.columns
    ]

    tests = tests[
        merge_columns
        + [
            "significant_bh_0_05",
        ]
        + optional_test_columns
    ].drop_duplicates(
        merge_columns,
        keep="last",
    )

    data = summary.merge(
        tests,
        on=merge_columns,
        how="left",
        validate="one_to_one",
    )

    data["significant_bh_0_05"] = (
        data["significant_bh_0_05"]
        .fillna(False)
        .astype(bool)
    )

    return (
        data.sort_values(
            [
                "dataset",
                "comparison",
                "lags",
                "horizon",
            ]
        )
        .reset_index(drop=True)
    )


def ordered_datasets(data):
    available_datasets = set(
        data["dataset"].unique()
    )

    known_datasets = [
        dataset
        for dataset in DATASET_ORDER
        if dataset in available_datasets
    ]

    remaining_datasets = sorted(
        available_datasets - set(DATASET_ORDER)
    )

    return known_datasets + remaining_datasets


def text_color_for_value(value, max_abs_delta):
    if abs(value) >= 0.56 * max_abs_delta:
        return "white"

    return "black"


def format_heatmap_annotation(delta, significant):
    sign = "+" if delta > 0 else ""

    if significant:
        return rf"${sign}{delta:.3f}^{{*}}$"

    return f"{sign}{delta:.3f}"


def create_family_vs_convolution_heatmap(
    data,
    regime_label,
    output_pdf,
    output_png,
    max_abs_delta,
):
    data = data[
        data["comparison"].isin(COMPARISON_ORDER)
    ].copy()

    datasets = ordered_datasets(data)

    if not datasets:
        raise ValueError(
            f"No valid datasets found for {regime_label}."
        )

    max_abs_delta = max(
        float(max_abs_delta),
        np.finfo(float).eps,
    )

    cmap = LinearSegmentedColormap.from_list(
        "family_advantage",
        [
            "#B2182B",
            "#F7F7F7",
            "#2166AC",
        ],
        N=256,
    )

    norm = TwoSlopeNorm(
        vmin=-max_abs_delta,
        vcenter=0.0,
        vmax=max_abs_delta,
    )

    figure_width = max(
        15.6,
        4.20 * len(datasets) + 1.15,
    )

    figure, axes = plt.subplots(
        nrows=len(COMPARISON_ORDER),
        ncols=len(datasets),
        figsize=(figure_width, 8.05),
        squeeze=False,
    )

    figure.subplots_adjust(
        left=0.105,
        right=0.905,
        top=0.895,
        bottom=0.135,
        wspace=0.28,
        hspace=0.34,
    )

    image = None

    for row_index, comparison in enumerate(COMPARISON_ORDER):
        for column_index, dataset in enumerate(datasets):
            axis = axes[row_index, column_index]

            subset = data.loc[
                (data["dataset"] == dataset)
                & (data["comparison"] == comparison)
            ].copy()

            lags = sorted(
                subset["lags"].unique()
            )

            horizons = sorted(
                subset["horizon"].unique()
            )

            if not lags or not horizons:
                axis.set_axis_off()
                continue

            matrix = np.full(
                (
                    len(lags),
                    len(horizons),
                ),
                np.nan,
                dtype=float,
            )

            annotations = np.full(
                (
                    len(lags),
                    len(horizons),
                ),
                "",
                dtype=object,
            )

            for lag_index, lag in enumerate(lags):
                for horizon_index, horizon in enumerate(horizons):
                    cell = subset.loc[
                        (subset["lags"] == lag)
                        & (subset["horizon"] == horizon)
                    ]

                    if cell.empty:
                        continue

                    record = cell.iloc[0]

                    delta = float(
                        record["mean_delta"]
                    )

                    significant = bool(
                        record["significant_bh_0_05"]
                    )

                    matrix[
                        lag_index,
                        horizon_index,
                    ] = delta

                    annotations[
                        lag_index,
                        horizon_index,
                    ] = format_heatmap_annotation(
                        delta=delta,
                        significant=significant,
                    )

            image = axis.imshow(
                matrix,
                cmap=cmap,
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )

            for lag_index in range(len(lags)):
                for horizon_index in range(len(horizons)):
                    value = matrix[
                        lag_index,
                        horizon_index,
                    ]

                    if np.isnan(value):
                        axis.text(
                            horizon_index,
                            lag_index,
                            "--",
                            ha="center",
                            va="center",
                            fontsize=11,
                            color="black",
                        )
                        continue

                    axis.text(
                        horizon_index,
                        lag_index,
                        annotations[
                            lag_index,
                            horizon_index,
                        ],
                        ha="center",
                        va="center",
                        fontsize=13.5,
                        fontweight="bold",
                        color=text_color_for_value(
                            value,
                            max_abs_delta,
                        ),
                    )

            axis.set_xticks(
                np.arange(len(horizons))
            )

            axis.set_xticklabels(
                horizons,
                fontsize=12,
            )

            axis.set_yticks(
                np.arange(len(lags))
            )

            axis.set_yticklabels(
                lags,
                fontsize=12,
            )

            axis.set_xlabel(
                "Forecasting horizon $H$",
                fontsize=12.5,
            )

            if column_index == 0:
                axis.set_ylabel(
                    "Context length $L$",
                    fontsize=12.5,
                    labelpad=6,
                )

            if row_index == 0:
                axis.set_title(
                    DATASET_LABELS.get(
                        dataset,
                        dataset,
                    ),
                    fontsize=13,
                    fontweight="bold",
                    pad=11,
                )

            axis.set_xticks(
                np.arange(
                    -0.5,
                    len(horizons),
                    1,
                ),
                minor=True,
            )

            axis.set_yticks(
                np.arange(
                    -0.5,
                    len(lags),
                    1,
                ),
                minor=True,
            )

            axis.grid(
                which="minor",
                color="black",
                linewidth=0.9,
            )

            axis.tick_params(
                which="minor",
                bottom=False,
                left=False,
            )

            for spine in axis.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

    if image is None:
        raise ValueError(
            f"No valid cells could be drawn for {regime_label}."
        )

    figure.text(
        0.024,
        0.661,
        COMPARISON_LABELS[
            "attention_minus_convolution"
        ],
        rotation=90,
        va="center",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    figure.text(
        0.024,
        0.349,
        COMPARISON_LABELS[
            "recurrent_minus_convolution"
        ],
        rotation=90,
        va="center",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    colorbar = figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.030,
        pad=0.025,
    )

    colorbar.set_label(
        r"$\Delta_{\mathrm{fam-conv}}="
        r"\overline{\mathrm{RMSE}}_{\mathrm{conv}}"
        r"-\overline{\mathrm{RMSE}}_{\mathrm{fam}}$"
        "\nPositive: family advantage | "
        "Negative: convolutional advantage",
        fontsize=11,
    )

    colorbar.ax.tick_params(
        labelsize=10.5
    )

    figure.suptitle(
        f"{regime_label} Paired RMSE Gaps by Temporal Context and Forecasting Horizon",
        fontsize=16,
        fontweight="bold",
        y=0.972,
    )

    figure.text(
        0.5,
        0.047,
        r"Cells show mean paired RMSE gaps. $^{*}$ indicates one-sided paired "
        r"Wilcoxon significance after Benjamini--Hochberg correction "
        r"at $\alpha=0.05$.",
        ha="center",
        va="center",
        fontsize=10.5,
    )

    output_pdf = Path(output_pdf)
    output_png = Path(output_png)

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_pdf,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def create_short_mid_heatmap(
    short_mid_data,
    output_pdf,
    output_png,
):
    max_abs_delta = float(
        np.nanmax(
            np.abs(
                short_mid_data[
                    "mean_delta"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    create_family_vs_convolution_heatmap(
        data=short_mid_data,
        regime_label="Short--Mid",
        output_pdf=output_pdf,
        output_png=output_png,
        max_abs_delta=max_abs_delta,
    )


def get_gap_record(
    data,
    dataset,
    lags,
    horizon,
    comparison,
):
    record = data.loc[
        (data["dataset"] == dataset)
        & (data["lags"] == lags)
        & (data["horizon"] == horizon)
        & (data["comparison"] == comparison)
    ]

    if record.empty:
        return None

    return record.iloc[0]


def format_gap_latex(record):
    if record is None:
        return "--"

    delta = float(
        record["mean_delta"]
    )

    ci95_delta = float(
        record["ci95_delta"]
    )

    significant = bool(
        record["significant_bh_0_05"]
    )

    sign = "+" if delta > 0 else ""
    marker = r"^{*}" if significant else ""

    return (
        rf"${sign}{delta:.4f} "
        rf"\pm {ci95_delta:.4f}{marker}$"
    )


def build_long_range_table(long_range_data):
    data = long_range_data[
        long_range_data["comparison"].isin(
            COMPARISON_ORDER
        )
    ].copy()

    rows = []

    for dataset in ordered_datasets(data):
        conditions = (
            data.loc[
                data["dataset"] == dataset,
                [
                    "lags",
                    "horizon",
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "lags",
                    "horizon",
                ]
            )
        )

        for _, condition in conditions.iterrows():
            lags = int(
                condition["lags"]
            )

            horizon = int(
                condition["horizon"]
            )

            attention_record = get_gap_record(
                data=data,
                dataset=dataset,
                lags=lags,
                horizon=horizon,
                comparison="attention_minus_convolution",
            )

            recurrent_record = get_gap_record(
                data=data,
                dataset=dataset,
                lags=lags,
                horizon=horizon,
                comparison="recurrent_minus_convolution",
            )

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(
                        dataset,
                        dataset,
                    ),
                    "lags": lags,
                    "horizon": horizon,
                    "configuration": f"({lags}, {horizon})",
                    "attention_gap": (
                        np.nan
                        if attention_record is None
                        else float(
                            attention_record[
                                "mean_delta"
                            ]
                        )
                    ),
                    "attention_ci95": (
                        np.nan
                        if attention_record is None
                        else float(
                            attention_record[
                                "ci95_delta"
                            ]
                        )
                    ),
                    "attention_significant": (
                        False
                        if attention_record is None
                        else bool(
                            attention_record[
                                "significant_bh_0_05"
                            ]
                        )
                    ),
                    "recurrent_gap": (
                        np.nan
                        if recurrent_record is None
                        else float(
                            recurrent_record[
                                "mean_delta"
                            ]
                        )
                    ),
                    "recurrent_ci95": (
                        np.nan
                        if recurrent_record is None
                        else float(
                            recurrent_record[
                                "ci95_delta"
                            ]
                        )
                    ),
                    "recurrent_significant": (
                        False
                        if recurrent_record is None
                        else bool(
                            recurrent_record[
                                "significant_bh_0_05"
                            ]
                        )
                    ),
                    "attention_latex": format_gap_latex(
                        attention_record
                    ),
                    "recurrent_latex": format_gap_latex(
                        recurrent_record
                    ),
                }
            )

    if not rows:
        raise ValueError(
            "No valid Long-Range conditions were found."
        )

    return pd.DataFrame(rows)


def export_long_range_latex_table(
    table,
    output_tex,
):
    output_tex = Path(output_tex)

    output_tex.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Paired family-level RMSE gaps relative to convolutional models in the Long-Range regime. The gap is defined as $\Delta_{\mathrm{fam-conv}}=\overline{\mathrm{RMSE}}_{\mathrm{conv}}-\overline{\mathrm{RMSE}}_{\mathrm{fam}}$, so positive values favor the compared family. Values are reported as mean gap $\pm$ $95\%$ confidence interval. Asterisks indicate significance after Benjamini--Hochberg correction.}",
        r"\label{tab:long_range_family_gaps}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{$(L,H)$} & \textbf{Attention $-$ Convolution} & \textbf{Recurrent $-$ Convolution} \\",
        r"\midrule",
    ]

    grouped = list(
        table.groupby(
            "dataset",
            sort=False,
        )
    )

    for group_index, (_, group) in enumerate(grouped):
        group = group.reset_index(drop=True)
        group_size = len(group)

        for row_index, row in group.iterrows():
            if row_index == 0 and group_size > 1:
                dataset_cell = (
                    rf"\multirow{{{group_size}}}{{*}}"
                    rf"{{{row['dataset_label']}}}"
                )
            elif row_index == 0:
                dataset_cell = row["dataset_label"]
            else:
                dataset_cell = ""

            lines.append(
                f"{dataset_cell} & "
                f"${row['configuration']}$ & "
                f"{row['attention_latex']} & "
                f"{row['recurrent_latex']} \\\\"
            )

        if group_index < len(grouped) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{0.93\linewidth}{\footnotesize Positive gaps indicate lower RMSE for the compared family.}",
            r"\end{table}",
        ]
    )

    output_tex.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )





