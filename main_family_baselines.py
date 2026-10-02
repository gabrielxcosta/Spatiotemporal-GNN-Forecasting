"""Execute exclusivamente os baselines do benchmark espaço-temporal.

Este executor reutiliza integralmente loaders, normalização, janelas, splits,
avaliação, métricas, figuras e convenções de diretório de ``main_family.py``.
Ele apenas restringe a seleção aos quatro baselines e aplica a expansão correta
para cada tipo:

* Persistence, SeasonalPersistence e LeastSquares são analíticos e
  determinísticos: executam uma vez, em ``seed_0``, e apenas com o primeiro
  valor de ``hidden`` usado na nomenclatura comum;
* xLSTM é treinável: executa todas as seeds e dimensões hidden solicitadas;
* ``noipj-long`` aplica-se somente ao xLSTM, por meio de ``xLSTM_w``. Os três
  métodos analíticos já não possuem projeção e usam ``long`` como referência.

Examples
--------
Executar todos os baselines no WikiMaths, com as duas representações::

    python3 -u main_family_baselines.py --dataset wikimaths \
        --regime all --representation all

Executar apenas os métodos analíticos, sem figuras::

    python3 -u main_family_baselines.py --dataset pemsbay \
        --models Persistence SeasonalPersistence LeastSquares \
        --regime short-mid --representation scalar --no-plots

Executar xLSTM com seeds 3 a 5::

    python3 -u main_family_baselines.py --dataset aqi36 --models xLSTM \
        --regime long --representation lagged --seed-start 3 --seed-end 5
"""

import argparse
import gc
import traceback

import torch

from main_family import (
    ANALYTIC_BASELINES,
    DATASETS,
    MODELS,
    configs_for,
    execute_job,
)
from utils.utilities import build_config_name


BASELINE_MODELS = tuple(
    name for name, (family, _, _) in MODELS.items() if family == "baseline"
)


def parse_args():
    """Leia e valide as opções do executor exclusivo de baselines.

    Returns
    -------
    argparse.Namespace
        Seleção de dataset, modelos, regimes, representações, seeds e
        hiperparâmetros compatível com ``main_family.configs_for``.

    Raises
    ------
    SystemExit
        Se o argparse receber escolhas inválidas, valores não positivos ou um
        intervalo de seeds incompleto/inconsistente.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=[*DATASETS, "all"], default="chickenpox"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=list(BASELINE_MODELS),
        help="Baselines a executar; por padrão, executa os quatro.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=[],
        help="Baselines a remover da seleção.",
    )
    parser.add_argument(
        "--regime",
        choices=["short-mid", "long", "noipj-long", "all"],
        default="all",
    )
    parser.add_argument(
        "--representation",
        choices=["scalar", "lagged", "all"],
        default="scalar",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Seeds do xLSTM a partir de zero; padrão: 10. Ignorado pelos analíticos.",
    )
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lags", type=int, nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge-drop", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    for name in ("hidden", "lags", "horizons"):
        values = getattr(args, name)
        if values is not None:
            if not values or min(values) < 1:
                parser.error(f"--{name} deve conter apenas inteiros positivos")
            setattr(args, name, list(dict.fromkeys(values)))
    for name in ("batch_size", "epochs", "warmup", "patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} deve ser positivo")
    if args.lr <= 0:
        parser.error("--lr deve ser positivo")
    if not 0 <= args.dropout < 1 or not 0 <= args.edge_drop < 1:
        parser.error("--dropout e --edge-drop devem pertencer a [0, 1)")
    if (args.seed_start is None) != (args.seed_end is None):
        parser.error("--seed-start e --seed-end devem ser fornecidos juntos")
    if args.seed_start is not None:
        if args.seeds is not None:
            parser.error("Use --seeds ou --seed-start/--seed-end, não ambos")
        if args.seed_start < 0 or args.seed_end < args.seed_start:
            parser.error("O intervalo deve satisfazer 0 <= seed-start <= seed-end")
    if args.seeds is None:
        args.seeds = 10
    if args.seeds < 1:
        parser.error("--seeds deve ser positivo")
    return args


def build_jobs(args):
    """Expanda a CLI em jobs, respeitando a natureza de cada baseline.

    Parameters
    ----------
    args : argparse.Namespace
        Opções validadas por :func:`parse_args`.

    Returns
    -------
    list[tuple]
        Tuplas ``(dataset, regime, representation, model_name, cfg, seed)``
        aceitas por ``main_family.execute_job``.

    Notes
    -----
    Baselines analíticos são omitidos de ``noipj-long`` e recebem somente a
    seed zero. O xLSTM usa todas as seeds solicitadas. A ordem é dataset,
    regime, representação, modelo, configuração e seed.
    """
    excluded = set(args.exclude_models)
    selected = [
        name for name in dict.fromkeys(args.models)
        if name not in excluded
    ]
    seeds = tuple(
        range(args.seed_start, args.seed_end + 1)
        if args.seed_start is not None
        else range(args.seeds)
    )
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    regimes = (
        ["short-mid", "long", "noipj-long"]
        if args.regime == "all"
        else [args.regime]
    )
    representations = (
        ["scalar", "lagged"]
        if args.representation == "all"
        else [args.representation]
    )

    jobs = []
    for dataset in datasets:
        for regime in regimes:
            for representation in representations:
                args.representation = representation
                for model_name in selected:
                    if model_name in ANALYTIC_BASELINES and regime == "noipj-long":
                        continue
                    model_seeds = (0,) if model_name in ANALYTIC_BASELINES else seeds
                    for cfg in configs_for(
                        dataset, regime, args, model_name=model_name
                    ):
                        jobs.extend(
                            (dataset, regime, representation, model_name, cfg, seed)
                            for seed in model_seeds
                        )
    return jobs


def main():
    """Execute sequencialmente os jobs de baseline e reporte falhas.

    Returns
    -------
    None
        Quando todos os jobs terminarem ou forem reaproveitados.

    Raises
    ------
    RuntimeError
        Se CUDA for solicitada e não estiver disponível.
    SystemExit
        Com código 1 quando pelo menos um job falhar.
    """
    args = parse_args()
    jobs = build_jobs(args)
    if not jobs:
        print("Nenhum job de baseline a executar após os filtros.")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível; use --device cpu para executar em CPU")

    failures = []
    print(f"Execução sequencial de baselines: {len(jobs)} jobs; dispositivo={args.device}")
    for ident, job in enumerate(jobs):
        print(
            f"Job {ident}: iniciado ({job[3]}, seed={job[5]}, "
            f"config={build_config_name(job[4])})"
        )
        try:
            execute_job(job)
        except Exception:
            failures.append(ident)
            print(f"Job {ident}: failed")
            traceback.print_exc()
        else:
            print(f"Job {ident}: completed")
        finally:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    print(f"Finalizado: {len(jobs) - len(failures)} concluídos; {len(failures)} falhas")
    if failures:
        print(f"Jobs com falha: {failures}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
