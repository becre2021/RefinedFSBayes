"""Save ID and OOD metrics and export mean +/- std tables without mixing runs."""
import argparse
import copy
import json
import fcntl
from datetime import datetime
import math
from pathlib import Path
import statistics

OOD_LABELS = {'svhn2': 'SVHN', 'lsun32': 'LSUN resized', 'place365': 'Places365'}
ID_LABELS = {'acc': 'Accuracy', 'nll': 'NLL', 'ece': 'ECE'}
METRIC_LABELS = {**ID_LABELS, **{key: f'AUROC {label}' for key, label in OOD_LABELS.items()}}



def format_evaluation_metrics(sample_count, id_metrics, ood_metrics):
    """One readable line per posterior-sample count, with plain numeric values."""
    fields = [f'{name}={float(id_metrics[name]):.6f}' for name in ID_LABELS]
    fields.extend(f'auroc_{name}={float(value):.6f}' for name, value in ood_metrics.items())
    return f'fs={sample_count} : ' + ', '.join(fields)


def entropy_auroc(inside, outside):
    """Match evaluate_v2's entropy score, retaining precision across seeds."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    def probabilities(outs):
        return outs.logit_list.log_softmax(dim=-1).exp().mean(dim=0).numpy()

    p_in, p_out = probabilities(inside), probabilities(outside)
    probabilities_all = np.concatenate([p_in, p_out])
    scores = -((probabilities_all + 1e-12) * np.log(probabilities_all + 1e-12)).sum(axis=1)
    labels = np.concatenate([np.zeros(len(p_in)), np.ones(len(p_out))])
    return float(roc_auc_score(labels, scores))


def metrics_for_sample(record, sample_count):
    """Read either legacy split metrics or the flat six-metric schema."""
    if record.get('schema_version') == 1:
        combined = dict(record.get('id_metrics', {}).get(str(sample_count), {}))
        combined.update({f'auroc_{name}': value for name, value in
                         record.get('metrics', {}).get(str(sample_count), {}).items()})
        return combined
    if record.get('schema_version') == 2:
        evaluations = [record, *record.get('additional_evaluations', [])]
        for evaluation in evaluations:
            if evaluation.get('nfsample') == sample_count:
                return evaluation['metrics']
    return {}


def flatten_result_record(record):
    """Use a flat metrics object; preserve all evaluated posterior sample counts."""
    result = copy.deepcopy(record)
    if result.get('schema_version') == 2:
        return result
    if result.get('schema_version') != 1:
        raise ValueError('Unsupported results schema')
    sample_counts = sorted({int(value) for section in ('metrics', 'id_metrics')
                            for value in result.get(section, {})})
    if not sample_counts or any(value < 1 for value in sample_counts):
        raise ValueError('Results must include a positive posterior sample count')
    evaluations = [{'nfsample': count, 'metrics': metrics_for_sample(record, count)}
                   for count in sample_counts]
    # The largest evaluated sample count is the main result. Other counts stay
    # available as explicitly labeled flat records for the summary reader.
    result.update(schema_version=2, **evaluations[-1])
    result.pop('id_metrics', None)
    if len(evaluations) > 1:
        result['additional_evaluations'] = evaluations[:-1]
    return result


def append_result_history(record, history_path):
    """Append one complete JSON record without replacing previous evaluations."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, allow_nan=False) + '\n'
    with history_path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()


def save_seed_results(path, configs, checkpoint, metrics, oodsets, places365_root, id_metrics=None, history_path=None,
                      table_seeds=None, table_nfsamples=None, table_ddof=0, table_filename='results_table.txt'):
    record = {
        'schema_version': 1,
        'metric': 'predictive_entropy_auroc',
        'seed': int(configs.exp.seeds),
        'checkpoint': str(Path(checkpoint).resolve()),
        'configs': configs,
        'evaluation': {'oodsets': list(oodsets), 'places365_root': places365_root},
        'metrics': metrics,
        'id_metrics': id_metrics or {},
    }
    record = flatten_result_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    history_record = {**record, 'result_file': str(path.resolve()),
                      'recorded_at': datetime.now().astimezone().isoformat(timespec='seconds')}
    append_result_history(history_record, history_path or path.parent / 'results.jsonl')
    table_dir = Path(history_path).parent if history_path else path.parent
    table_dir.mkdir(parents=True, exist_ok=True)
    # Serialize scans as well as writes so concurrent seeds cannot append stale snapshots.
    with (table_dir / '.results_table.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        paths = []
        key = experiment_key(record)
        for candidate in sorted(set(path.parent.glob('*.ood.json')) | {path}):
            other = json.loads(candidate.read_text())
            if experiment_key(other) == key:
                paths.append(candidate)
        rows = summarize(paths, table_seeds or [1111, 1112, 1113],
                         table_nfsamples or [record['nfsample']], table_ddof)
        if not all(row['status'] == 'complete' for row in rows):
            return None
        write_tables(rows, table_dir / table_filename, append=True, run_label=path.parent.name)
    return table_dir / table_filename


def experiment_key(record):
    configs = copy.deepcopy(record['configs'])
    for key in ('seeds', 'v', 'nexp'):
        configs.get('exp', {}).pop(key, None)
    return json.dumps({'configs': configs, 'evaluation': record['evaluation'],
                       'metric': record['metric']}, sort_keys=True)


def trajectory_description(configs):
    """Describe the schedule used by the local BMA-FBNN trainers."""
    train = configs['train']
    if not configs['model']['modelname'].startswith('bmafb'):
        return 'trajectory=not applicable'
    if train.get('prior_distribution', 'trajectory') == 'fixed':
        return 'trajectory=not applicable (fixed prior)'
    epochs, ratio = train.get('nepoch'), train.get('nepochratio')
    if epochs is None or ratio is None or epochs <= 0:
        return 'trajectory=unspecified'
    end = int(epochs * ratio)
    start = end - int(epochs * 0.1)
    import numpy as np
    scheduled = np.linspace(start, end, 4, dtype=int).tolist()
    return (f'trajectory={start / epochs:.2f}:{end / epochs:.2f}, '
            f'nepochratio={ratio:g}, bayes_start_epoch={end}, '
            f'trajectory_scheduled_epochs={scheduled}, trajectory_scheduled_count=4')


def summarize(paths, expected_seeds, sample_counts, ddof=0):
    if len(set(expected_seeds)) != len(expected_seeds) or len(expected_seeds) < 2:
        raise ValueError('Expected seeds must contain at least two distinct values')
    if ddof not in (0, 1):
        raise ValueError('ddof must be 0 or 1')
    groups = {}
    for path in paths:
        record = json.loads(Path(path).read_text())
        if record.get('schema_version') not in (1, 2) or record.get('metric') != 'predictive_entropy_auroc':
            raise ValueError(f'Unsupported results format: {path}')
        seed = record['seed']
        if seed not in expected_seeds:
            raise ValueError(f'Unexpected seed {seed}: {path}')
        key = experiment_key(record)
        group = groups.setdefault(key, {})
        if seed in group:
            raise ValueError(f'Duplicate seed {seed} for the same experiment: {path}')
        group[seed] = record

    rows = []
    for key, records in sorted(groups.items()):
        configs = next(iter(records.values()))['configs']
        for nfsample in sample_counts:
            row = {
                'dataset': configs['dataset']['trainset'],
                'model': configs['model']['modelname'],
                'scalelogstd': configs['model']['scalelogstd'],
                'reglambda': configs['train']['reglambda'],
                'mixalpha': configs['train']['mixalpha'],
                'priorstdout': configs['model']['priorstdout'],
                'nfsample': nfsample,
                'ntrainset': configs['dataset'].get('ntrainset', 'unspecified'),
                'nepoch': configs['train'].get('nepoch', 'unspecified'),
                'trajectory_description': trajectory_description(configs),
                'expected_seeds': ','.join(map(str, expected_seeds)),
                'ddof': ddof,
                'experiment_json': key,
                'seed_values': {seed: {} for seed in expected_seeds},
            }
            complete = True
            for name in METRIC_LABELS:
                values = []
                for seed in expected_seeds:
                    metric_name = name if name in ID_LABELS else f'auroc_{name}'
                    value = metrics_for_sample(records.get(seed, {}), nfsample).get(metric_name)
                    if value is not None:
                        value = float(value)
                        if not math.isfinite(value) or value < 0 or (name != 'nll' and value > 1):
                            label = 'AUROC' if name in OOD_LABELS else name
                            raise ValueError(f'Invalid {label} for seed {seed}: {value}')
                        values.append(value)
                        row['seed_values'][seed][name] = value
                row[f'{name}_n'] = len(values)
                if len(values) == len(expected_seeds):
                    mean = statistics.mean(values)
                    std = statistics.pstdev(values) if ddof == 0 else statistics.stdev(values)
                    row[f'{name}_mean'] = mean
                    row[f'{name}_std'] = std
                    row[f'{name}_cell'] = f'{mean:.3f} ± {std:.3f}'
                else:
                    complete = False
                    row[f'{name}_mean'] = ''
                    row[f'{name}_std'] = ''
                    row[f'{name}_cell'] = f'incomplete ({len(values)}/{len(expected_seeds)})'
            row['status'] = 'complete' if complete else 'incomplete'
            rows.append(row)
    return rows


def write_tables(rows, output_prefix, method_label=None, append=False, run_label=None):
    """Write each seed followed by a final, copyable LaTeX mean/std row."""
    if not rows:
        raise ValueError('No experiment results found')
    output = Path(output_prefix).with_suffix('.txt')
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '% Columns: Method/seed & epochs & acc & nll & ece & auroc-svhn & auroc-lsun & auroc-place365',
        f"% Statistics: mean and standard deviation across seeds; ddof={rows[0]['ddof']}.",
    ]
    if run_label:
        lines.insert(0, f'% Run: {run_label}; written {datetime.now().astimezone().isoformat(timespec="seconds")}')
    for row in rows:
        model = row['model']
        if method_label:
            method = method_label
        elif model.startswith('bmafb'):
            method = 'BMA-FBNN-ST' if model.endswith('st') else 'BMA-FBNN'
        elif model in ('resnet18', 'resnet50', 'resnet20'):
            method = 'MAP'
        else:
            method = model
        method = method.replace('_', r'\_')
        lines.extend([
            '',
            '##----------------------------------------------------------------------------------------------------------#',
            f"% {row['dataset']}/{model}; logstd={row['scalelogstd']}, "
            f"reg={row['reglambda']}, mix={row['mixalpha']}, prior={row['priorstdout']}, "
            f"samples={row['nfsample']}; ntrainset={row['ntrainset']}, epochs={row['nepoch']}; "
            f"{row['trajectory_description']}",
        ])
        if row['status'] != 'complete':
            lines.append('% Incomplete cells are shown as --; they are not averaged.')
        for seed, values in row['seed_values'].items():
            cells = [f'${values[name]:.3f}$' if name in values else '--' for name in METRIC_LABELS]
            lines.append(' & '.join([f'{method} (seed={seed})', str(row['nepoch']), *cells]) + r' \\')
        summary = [
            f"${row[f'{name}_mean']:.3f} \\pm {row[f'{name}_std']:.3f}$"
            if row[f'{name}_mean'] != '' else '--'
            for name in METRIC_LABELS
        ]
        lines.append(' & '.join([method, str(row['nepoch']), *summary]) + r' \\')
    text = '\n'.join(lines) + '\n'
    with output.open('a+' if append else 'w') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if append and run_label:
            # The same study may be reported after its model and again at study end.
            stream.seek(0)
            existing = stream.read()
            header, body = text.split('\n', 1)
            label_prefix = header.split('; written ', 1)[0] + '; written '
            for block in existing.split('% Run: ')[1:]:
                previous_header, _, previous_body = block.partition('\n')
                if ('% Run: ' + previous_header).startswith(label_prefix) and previous_body.startswith(body):
                    return text
            stream.seek(0, 2)
        if append and stream.tell() > 0:
            stream.write('\n')
        stream.write(text)
        stream.flush()
    return text



def write_dataset_tables(rows, output_prefix, **options):
    """Keep each dataset's accumulated table in its own result directory."""
    grouped = {}
    for row in rows:
        dataset = row['dataset']
        if dataset in ('.', '..') or Path(dataset).name != dataset:
            raise ValueError(f'Invalid dataset name: {dataset}')
        grouped.setdefault(dataset, []).append(row)
    prefix = Path(output_prefix)
    outputs = {}
    for dataset, dataset_rows in grouped.items():
        destination = prefix.parent / f'runs_{dataset}' / prefix.name
        outputs[destination.with_suffix('.txt')] = write_tables(dataset_rows, destination, **options)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--expected-seeds', nargs='+', type=int, default=[1111, 1112, 1113])
    parser.add_argument('--nfsamples', nargs='+', type=int, default=[20])
    parser.add_argument('--ddof', type=int, choices=[0, 1], default=0)
    parser.add_argument('--output', type=Path, default=Path('results/statistics/results_table'))
    parser.add_argument('--method-label', help='optional paper-table label, e.g. MAP')
    parser.add_argument('--append', action='store_true', help='accumulate this study without replacing earlier results')
    parser.add_argument('--run-label', help='study identifier, usually the source log directory')
    parser.add_argument('--split-by-dataset', action='store_true',
                        help='write each table to runs_<dataset>/results_table.txt')
    args = parser.parse_args()
    try:
        rows = summarize(args.results, args.expected_seeds, args.nfsamples, args.ddof)
        options = dict(method_label=args.method_label, append=args.append, run_label=args.run_label)
        if args.split_by_dataset:
            outputs = write_dataset_tables(rows, args.output, **options)
        else:
            outputs = {args.output.with_suffix('.txt'): write_tables(rows, args.output, **options)}
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    for path, text in outputs.items():
        print(f'Wrote {path}')
        print(text, end='')


if __name__ == '__main__':
    main()
