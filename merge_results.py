"""Merge local experiment metrics without averaging different configurations or runs."""
import argparse
import csv
import json
from pathlib import Path

from ood_results import metrics_for_sample

METRICS = ('acc', 'nll', 'ece', 'auroc_svhn2', 'auroc_lsun32', 'auroc_place365')
FIELDS = ('dataset', 'method', 'model', 'seed', 'version', 'split', 'samples', 'eval_batch',
          'logstd', 'reglambda', 'priorstdout', 'mixalpha', 'nepochratio',
          'lr', 'epochs', 'steps', 'nbatch', 'nfbatch', 'nctset',
          *METRICS, 'source', 'checkpoint', 'configs_json')


def make_row(configs, metrics, source, checkpoint, split, samples):
    dataset, model, train, exp = (configs.get(k, {}) for k in ('dataset', 'model', 'train', 'exp'))
    name = model.get('modelname', '')
    infer = model.get('infer', '')
    if infer == 'map' or name in ('resnet18', 'resnet50'):
        method = 'MAP'
    elif name.endswith('st'):
        method = 'R-FVI (fully stochastic)'
    else:
        method = 'R-FVI (last layer)'
    return dict(dataset=dataset.get('trainset', ''), method=method, model=name,
                seed=exp.get('seeds', ''), version=exp.get('v', ''), split=split,
                samples=samples, eval_batch='', logstd=model.get('scalelogstd', ''),
                priorstdout=model.get('priorstdout', ''),
                epochs=train.get('nepoch', ''), steps=train.get('nstep', ''),
                **{key: train.get(key, '') for key in
                   ('reglambda', 'mixalpha', 'nepochratio', 'lr', 'nbatch', 'nfbatch', 'nctset')},
                **{key: float(metrics[key]) if metrics.get(key) is not None else '' for key in METRICS},
                source=str(source), checkpoint=str(checkpoint),
                configs_json=json.dumps(configs, sort_keys=True))


def collect_rows(root, samples):
    rows = []
    # JSONL contains historical snapshots of the same results; use current JSON only.
    paths = sorted(set(root.rglob('results.json')) | set(root.rglob('*.ood.json')))
    seen = set()
    for path in paths:
        record = json.loads(path.read_text())
        if record.get('metric') != 'predictive_entropy_auroc':
            continue
        model_config = record['configs']['model']
        evaluation_samples = 1 if model_config.get('infer') == 'map' or model_config['modelname'] in ('resnet18', 'resnet50') else samples
        metrics = metrics_for_sample(record, evaluation_samples)
        if not metrics:
            print(f'Skipping {path}: no evaluation with {samples} samples')
            continue
        checkpoint = record.get('checkpoint', '')
        # The same checkpoint may have been re-evaluated into multiple result files.
        eval_batch = record.get('evaluation', {}).get('signature', {}).get('batch', '')
        identity = (checkpoint or str(path), evaluation_samples, eval_batch, json.dumps(metrics, sort_keys=True))
        if identity in seen:
            continue
        seen.add(identity)
        row = make_row(record['configs'], metrics, path, checkpoint, 'test', evaluation_samples)
        row['eval_batch'] = eval_batch
        row['seed'] = record['seed']
        rows.append(row)
    # ViT trainers currently save validation metrics in their best checkpoints.
    # Load only trusted local checkpoints produced by these experiments.
    checkpoints = sorted((root / 'dtd_tinyimagenet').rglob('*.pth.tar'))
    if checkpoints:
        import torch
        for path in checkpoints:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if 'acc_val' in saved and 'nll_val' in saved:
                configs = saved['configs']
                if configs['dataset']['trainset'] == 'dtd':
                    rows.append(make_row(configs, {'acc': saved['acc_val'], 'nll': saved['nll_val']},
                                         path, path, 'validation', configs['train'].get('nfsample', '')))
            del saved
    return sorted(rows, key=lambda r: (r['dataset'], r['method'], str(r['version']), str(r['seed']), r['source']))


def write_comparison(rows, destination):
    """Write readable side-by-side comparisons, separated by dataset and split."""
    labels = {
        'method': 'Method', 'model': 'Model', 'seed': 'Seed', 'samples': 'Samples', 'eval_batch': 'Eval. batch',
        'logstd': 'logstd', 'reglambda': 'lambda', 'priorstdout': 'tau',
        'mixalpha': 'mixalpha', 'nepochratio': 'Bayesian start', 'lr': 'LR',
        'epochs': 'Epochs', 'steps': 'Steps', 'nbatch': 'Train batch',
        'nfbatch': 'Effective batch', 'nctset': 'Context groups',
        'acc': 'Accuracy ↑', 'nll': 'NLL ↓', 'ece': 'ECE ↓',
        'auroc_svhn2': 'AUROC SVHN ↑', 'auroc_lsun32': 'AUROC LSUN ↑',
        'auroc_place365': 'AUROC Places365 ↑',
    }
    config_keys = [key for key in labels if key not in METRICS]

    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', ' ')

    lines = ['# Experiment comparison', '',
             'Each row is a separate run. Configurations are taken from saved results/checkpoints. '
             'Validation and test results are shown separately. No runs are averaged.', '']
    for dataset, split in sorted({(r['dataset'], r['split']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['split']) == (dataset, split)]
        lines.extend([f'## {dataset.upper()} — {split}', ''])
        common = [key for key in config_keys if len({str(r[key]) for r in group}) == 1 and group[0][key] != '']
        varying = [key for key in config_keys if len({str(r[key]) for r in group}) > 1]
        if common:
            lines.extend(['Shared settings: ' + '; '.join(f'{labels[key]} = {cell(group[0][key])}' for key in common) + '.', ''])
        if split == 'validation':
            lines.extend(['Saved best-validation-NLL checkpoints. These are validation metrics; test/OOD metrics, when evaluated, appear in a separate test table.', ''])
        metrics = [key for key in METRICS if any(r[key] != '' for r in group)]
        columns = varying + metrics
        lines.extend(['| Run version | ' + ' | '.join(labels[key] for key in columns) + ' |',
                      '| --- | ' + ' | '.join('---' for _ in columns) + ' |'])
        for row in group:
            values = [f"{row[key]:.4f}" if key in metrics and row[key] != '' else cell(row[key]) if row[key] != '' else '--' for key in columns]
            lines.append('| ' + str(row['version']) + ' | ' + ' | '.join(values) + ' |')
        lines.extend(['', 'Sources:', ''])
        for row in group:
            lines.append(f"- Run {row['version']}: `{row['source']}`")
        lines.append('')
    (destination / 'comparison.md').write_text('\n'.join(lines) + '\n')


def latex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                    '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def write_latex_comparison(rows, destination):
    """Export booktabs tables for inclusion in a paper and a standalone document."""
    config_labels = {
        'method': 'Method', 'model': 'Model', 'seed': 'Seed', 'samples': 'Samples', 'eval_batch': 'Eval. batch',
        'logstd': 'logstd', 'reglambda': r'$\lambda$', 'priorstdout': r'$\tau$',
        'mixalpha': r'$\alpha$', 'nepochratio': 'Bayesian start', 'lr': 'LR',
        'epochs': 'Epochs', 'steps': 'Steps', 'nbatch': 'Train batch',
        'nfbatch': 'Effective batch', 'nctset': 'Context groups',
    }
    metric_labels = {'acc': r'Acc.$\uparrow$', 'nll': r'NLL$\downarrow$',
                     'ece': r'ECE$\downarrow$', 'auroc_svhn2': r'SVHN$\uparrow$',
                     'auroc_lsun32': r'LSUN$\uparrow$', 'auroc_place365': r'Places365$\uparrow$'}

    def format_config(key, value):
        if value == '':
            return '--'
        if key == 'reglambda':
            # Show small regularizers consistently in scientific notation.
            if 0 < abs(float(value)) < 0.01:
                mantissa, exponent = f'{float(value):.8e}'.split('e')
                coefficient = float(mantissa)
                prefix = '' if coefficient == 1 else f'{coefficient:g}' + r'\times '
                return '$' + prefix + '10^{' + str(int(exponent)) + '}$'
        if key not in ('method', 'model'):
            try:
                return latex_escape(f'{float(value):g}')
            except (ValueError, TypeError):
                pass
        return latex_escape(value)

    lines = [r'% Requires \usepackage{booktabs,graphicx}.',
             '% Each run is separate. No averaging across configurations or repeated seeds.', '']
    for dataset, split in sorted({(r['dataset'], r['split']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['split']) == (dataset, split)]
        common = [key for key in config_labels if len({format_config(key, r[key]) for r in group}) == 1 and group[0][key] != '']
        varying = [key for key in config_labels if len({format_config(key, r[key]) for r in group}) > 1]
        metrics = [key for key in METRICS if any(r[key] != '' for r in group)]
        columns = varying + metrics
        caption = latex_escape(dataset.upper() + ' (' + split + ')') + '. '
        caption += '; '.join(config_labels[key] + ' = ' + format_config(key, group[0][key]) for key in common) + '.'
        if split == 'validation':
            caption += ' Best-validation-NLL checkpoints; validation results are separate from test and OOD evaluation.'
        elif any(key.startswith('auroc_') for key in metrics):
            caption += ' SVHN, LSUN, and Places365 columns report OOD AUROC.'
        caption += ' Values are proportions, not percentages.'
        lines.extend([r'\begin{table}[htbp]', r'\centering', r'\small',
                      r'\caption{' + caption + '}',
                      r'\label{tab:comparison-' + latex_escape(dataset + '-' + split) + '}',
                      r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{l' + 'r' * len(columns) + '}',
                      r'\toprule', 'Run & ' + ' & '.join(config_labels.get(key, metric_labels.get(key)) for key in columns) + r' \\',
                      r'\midrule'])
        for index, row in enumerate(group, 1):
            lines.append('% Run R' + str(index) + ': version=' + str(row['version']) + '; source=' + row['source'].replace('\n', ' '))
            cells = [f'{row[key]:.4f}' if key in metrics and row[key] != '' else format_config(key, row[key]) for key in columns]
            lines.append('R' + str(index) + ' & ' + ' & '.join(cells) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}', ''])
    content = '\n'.join(lines) + '\n'
    (destination / 'comparison_tables.tex').write_text(content)
    preamble = '\n'.join([r'\documentclass{article}', r'\usepackage[margin=1.5cm,landscape]{geometry}',
                          r'\usepackage{booktabs,graphicx}', r'\begin{document}', ''])
    (destination / 'comparison.tex').write_text(preamble + content + '\\end{document}\n')


def write_merged(rows, destination):
    destination.mkdir(parents=True, exist_ok=True)
    write_comparison(rows, destination)
    write_latex_comparison(rows, destination)
    with (destination / 'results_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['% One row per run; different configurations and repeated seeds are not averaged.',
             '% -- means unavailable. Validation metrics are not test metrics.',
             '% Columns: Method/seed & epochs or steps & acc & nll & ece & auroc-svhn & auroc-lsun & auroc-place365']
    for row in rows:
        lines.extend(['', '##----------------------------------------------------------------------------------------------------------#',
                      '% ' + '; '.join(f'{key}={row[key]}' for key in FIELDS
                                      if key not in (*METRICS, 'source', 'checkpoint', 'configs_json') and row[key] != ''),
                      '% source: ' + row['source'],
                      '% checkpoint: ' + row['checkpoint']])
        budget = f"{row['epochs']} epochs" if row['epochs'] != '' else f"{row['steps']} steps"
        cells = [f"{row['method']} (seed={row['seed']})", budget]
        cells.extend(f'${row[key]:.3f}$' if row[key] != '' else '--' for key in METRICS)
        lines.append(' & '.join(cells) + r' \\')
    (destination / 'results_table.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('results/merged'))
    parser.add_argument('--nfsamples', type=int, default=10, help='ResNet posterior samples to report')
    args = parser.parse_args()
    if args.nfsamples < 1:
        parser.error('--nfsamples must be positive')
    rows = collect_rows(args.results_root, args.nfsamples)
    if not rows:
        parser.error('No saved metrics found')
    write_merged(rows, args.output)
    for dataset in sorted({row['dataset'] for row in rows}):
        if not dataset or Path(dataset).name != dataset or dataset in ('.', '..'):
            raise ValueError(f'Invalid dataset name: {dataset}')
        subset = [row for row in rows if row['dataset'] == dataset]
        write_merged(subset, args.output / dataset)
        for batch in sorted({row['eval_batch'] for row in subset if row['eval_batch'] != ''}):
            batch_rows = [row for row in subset if row['eval_batch'] == batch]
            write_merged(batch_rows, args.output / dataset / f'batch{int(batch)}')
        print(f'{args.output / dataset / "results_table.txt"}: {len(subset)} runs')
    print(f'{args.output / "results_table.txt"}: {len(rows)} runs total')


if __name__ == '__main__':
    main()
