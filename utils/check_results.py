"""Audita somente metrics.json; veja utils/check_results.md para exemplos."""

import argparse
import ast
from itertools import product
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def model_registry():
    """Lê o registro do runner sem importar torch ou executar treinamento."""
    tree = ast.parse((PROJECT_ROOT / 'main_family.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'MODELS'
            for target in node.targets
        ):
            return {name: spec for name, spec in ast.literal_eval(node.value).items()
                    if spec[0] != 'baseline'}
    raise ValueError('Registro MODELS não encontrado em main_family.py')


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_metrics(path, seed, config):
    try:
        with path.open() as stream:
            data = json.load(stream)
    except FileNotFoundError:
        return 'missing', 'metrics.json ausente'
    except (OSError, ValueError) as exc:
        return 'invalid', str(exc)
    errors = []
    if not isinstance(data, dict):
        return 'invalid', 'JSON deve ser um objeto'
    if type(data.get('seed')) is not int or data['seed'] != seed:
        errors.append('seed divergente ou ausente')
    actual = data.get('config')
    if not isinstance(actual, dict):
        errors.append('config ausente ou inválida')
    else:
        for key, expected in config.items():
            if not finite_number(actual.get(key)) or actual[key] != expected:
                errors.append(f'config.{key}: esperado {expected}, encontrado {actual.get(key)!r}')
    for key in ('test_mse', 'test_rmse', 'test_mae', 'test_mape',
                'test_r2_global', 'test_r2_mean_horizon', 'runtime_sec'):
        value = data.get(key)
        if not finite_number(value):
            errors.append(f'{key} ausente, não numérico ou não finito')
        elif key not in ('test_r2_global', 'test_r2_mean_horizon') and value < 0:
            errors.append(f'{key} negativo')
    epochs = data.get('epochs_ran')
    if type(epochs) is not int or epochs < 0:
        errors.append('epochs_ran deve ser inteiro não negativo')
    values = data.get('test_r2_per_horizon')
    if (not isinstance(values, list) or len(values) != config['horizon']
            or not all(finite_number(value) for value in values)):
        errors.append('test_r2_per_horizon inválido ou tamanho diferente do horizonte')
    return ('invalid', '; '.join(errors)) if errors else ('ok', '')


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('deve ser positivo')
    return value


def check_combination(task):
    root, regime, model, cfg, seeds = task
    tag = (f"l{cfg['lags']}_h{cfg['horizon']}_hid{cfg['hidden']}_lr{cfg['lr']}"
           f"_bs{cfg['batch_size']}_drop{cfg['dropout']}_edrop{cfg['edge_drop']}")
    groups = {'missing': [], 'invalid': []}
    for seed in seeds:
        path = root / model / tag / f'seed_{seed}' / 'metrics.json'
        status, _ = validate_metrics(path, seed, cfg)
        if status != 'ok':
            groups[status].append(seed)
    if not any(groups.values()):
        return None
    labels = {'missing': 'ausente', 'invalid': 'inválido'}
    return (f"{root} [{regime}] {model} contexto={cfg['lags']} "
            f"horizonte={cfg['horizon']} hidden={cfg['hidden']}: "
            + '; '.join(f"metrics.json {labels[status]} sementes={values}"
                        for status, values in groups.items() if values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', help='Pastas results_*; aceita pastas ausentes')
    parser.add_argument('--family', choices=['all', 'recurrent', 'convolutional', 'attention'], default='all')
    parser.add_argument('--architectures', nargs='+', help='Substitui a seleção por família')
    parser.add_argument('--seeds', type=positive, default=10, help='Quantidade: 0 até N-1')
    parser.add_argument('--seed-ids', type=int, nargs='+', help='Sementes explícitas')
    parser.add_argument('--hidden', type=positive, nargs='+', default=[32, 64])
    parser.add_argument('--contexts', type=positive, nargs='+')
    parser.add_argument('--horizons', type=positive, nargs='+')
    parser.add_argument('--regime', choices=['auto', 'short-mid', 'long', 'noipj-long'], default='auto')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=positive, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--edge-drop', type=float, default=0.1)
    parser.add_argument('--workers', type=positive, default=os.cpu_count()-1,
                        help='Número de processos (padrão: até 8)')
    args = parser.parse_args(argv)
    roots = args.roots or sorted(p for p in PROJECT_ROOT.glob('results_*') if p.is_dir())
    if not roots:
        parser.error('Nenhuma pasta results_* encontrada; informe --roots')
    registry = model_registry()
    if args.architectures:
        unknown = sorted(set(args.architectures) - registry.keys())
        if unknown:
            parser.error('Arquiteturas não suportadas (baselines excluídas): ' + ', '.join(unknown))
    models = sorted(set(args.architectures or [name for name, spec in registry.items()
                                             if args.family == 'all' or spec[0] == args.family]))
    seeds = sorted(set(args.seed_ids if args.seed_ids is not None else range(args.seeds)))
    tasks = []
    for root in dict.fromkeys(roots):
        regime = args.regime
        if regime == 'auto':
            regime = 'noipj-long' if root.name.endswith('_noipj_long') else 'long' if root.name.endswith('_long') else 'short-mid'
        england = 'englandcovid' in root.name.lower()
        contexts = sorted(set(args.contexts or ([2, 4, 8, 12] if regime == 'short-mid' else [20 if england else 50])))
        horizons = sorted(set(args.horizons or ([1, 5, 10] if regime == 'short-mid' else [10 if england else 20])))
        for model, context, horizon, hidden in product(models, contexts, horizons, sorted(set(args.hidden))):
            cfg = dict(lags=context, horizon=horizon, hidden=hidden, lr=args.lr,
                       batch_size=args.batch_size, dropout=args.dropout, edge_drop=args.edge_drop)
            tasks.append((root, regime, model, cfg, seeds))
    has_issues = False
    with mp.Pool(processes=min(args.workers, len(tasks))) as pool:
        for line in pool.imap(check_combination, tasks, chunksize=1):
            if line is not None:
                has_issues = True
                print(line, flush=True)
    if not has_issues:
        print('Tudo OK com os metrics.json.')
    return int(has_issues)


if __name__ == '__main__':
    raise SystemExit(main())
