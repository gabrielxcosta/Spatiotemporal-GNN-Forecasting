# -*- coding: utf-8 -*-
"""Junta os doze plots de utils.networks na figura final do benchmark."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

from utils.networks.common import FIGURES, GEPHI_CMAP


DATASETS = {
    "chickenpox": ("utils.networks.chickenpox", "HungaryNetwork", "Chickenpox"),
    "wikimath": ("utils.networks.wikimaths", "WikiMathNetwork", "WikiMath"),
    "england_covid": ("utils.networks.englandcovid", "EnglandCovidNetwork", "EnglandCOVID"),
    "montevideo": ("utils.networks.montevideobus", "MontevideoNetwork", "MontevideoBus"),
    "pedalme": ("utils.networks.pedalme", "PedalMeNetwork", "PedalMe"),
    "twitter_rg17": ("utils.networks.twittertennis_rg17", "TwitterTennisRG17Network", "TwitterTennis RG17"),
    "twitter_uo17": ("utils.networks.twittertennis_uo17", "TwitterTennisUO17Network", "TwitterTennis UO17"),
    "pems_bay": ("utils.networks.pemsbay", "PeMSBayNetwork", "PeMS-Bay"),
    "aqi36": ("utils.networks.aqi36", "AQI36Network", "AQI36"),
    "aqi437": ("utils.networks.aqi437", "AQI437Network", "AQI437"),
    "rio_negro": ("utils.networks.rionegro", "RioNegroNetwork", "RioNegro"),
    "grid2op": ("utils.networks.grid2op", "Grid2OpNetwork", "Grid2Op IEEE-14"),
}


def _network(dataset_name, dynamic_view="aggregate", snapshot_id=0):
    module_name, class_name, _ = DATASETS[dataset_name]
    cls = getattr(importlib.import_module(module_name), class_name)
    if dataset_name in {"england_covid", "twitter_rg17", "twitter_uo17"}:
        return cls(dynamic_view=dynamic_view, snapshot_id=snapshot_id)
    return cls()


class MultiDatasetPlot:
    def __init__(self, dynamic_view="aggregate", snapshot_id=0):
        self.dynamic_view = dynamic_view
        self.snapshot_id = snapshot_id
        self.cmap = GEPHI_CMAP
        self.norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    def plot_all(self, output=FIGURES / "dataset_networks.pdf", save_png=True):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 3, figsize=(7.48, 8.8))

        for index, (dataset_name, (_, _, title)) in enumerate(DATASETS.items()):
            ax = axes.flat[index]
            network = _network(dataset_name, self.dynamic_view, self.snapshot_id)
            network.plot_network(ax=ax)
            ax.set_title(f"({chr(97 + index)}) {title}",
                         fontsize=8, fontweight="normal", pad=2)
            ax.axis("off")

        color_axis = fig.add_axes([.28, .025, .44, .012])
        colorbar = fig.colorbar(
            ScalarMappable(norm=self.norm, cmap=self.cmap),
            cax=color_axis, orientation="horizontal")
        colorbar.outline.set_visible(False)
        for spine in colorbar.ax.spines.values():
            spine.set_visible(False)
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label("Normalized betweenness centrality", fontsize=8)

        fig.subplots_adjust(left=.025, right=.985, top=.985, bottom=.065,
                            wspace=.08, hspace=.14)
        fig.savefig(output, bbox_inches="tight")
        if save_png:
            fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output


def save_full_network_figure(output=FIGURES / "dataset_networks.pdf", save_png=True,
                             dynamic_view="aggregate", snapshot_id=0):
    return MultiDatasetPlot(dynamic_view, snapshot_id).plot_all(output, save_png)


def save_debug_plot(dataset_name, figsize=(12, 8),
                    output_dir=FIGURES / "network_debug",
                    dynamic_view="aggregate", snapshot_id=0):
    if dataset_name not in DATASETS:
        raise KeyError(f"dataset inválido: {dataset_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    network = _network(dataset_name, dynamic_view, snapshot_id)
    _, _, title = DATASETS[dataset_name]
    fig, ax = plt.subplots(figsize=figsize)
    network.plot_network(ax=ax)
    ax.set_title(title, fontsize=12, fontweight="normal")
    output = output_dir / f"{dataset_name}.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_all_debug_plots(output_dir=FIGURES / "network_debug",
                         dynamic_view="aggregate", snapshot_id=0):
    return [save_debug_plot(name, output_dir=output_dir,
                            dynamic_view=dynamic_view, snapshot_id=snapshot_id)
            for name in DATASETS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--dynamic-view", choices=["aggregate", "snapshot"],
                        default="aggregate")
    parser.add_argument("--snapshot-id", type=int, default=0)
    args = parser.parse_args()
    if args.dataset:
        save_debug_plot(args.dataset, dynamic_view=args.dynamic_view,
                        snapshot_id=args.snapshot_id)
    elif args.debug:
        save_all_debug_plots(dynamic_view=args.dynamic_view,
                             snapshot_id=args.snapshot_id)
    else:
        save_full_network_figure(dynamic_view=args.dynamic_view,
                                 snapshot_id=args.snapshot_id)


if __name__ == "__main__":
    main()
