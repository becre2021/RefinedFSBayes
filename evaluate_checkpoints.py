"""Evaluate trusted local MAP/last-layer checkpoints without retraining."""
import argparse
import copy
import gc
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from addict import Dict
from sklearn.metrics import roc_auc_score

from evaluate_v2 import get_ece, get_ll
from ood_results import save_seed_results


class ImageDirectory(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(p for p in Path(root).rglob('*') if p.is_file() and
                            p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        if not self.paths:
            raise FileNotFoundError(f'No images in {root}')
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB')), -1


def make_model(configs):
    name = configs.model.modelname
    if name.startswith('vit'):
        if name != 'vitbp16r224':
            raise ValueError(f'Unsupported ViT architecture: {name}')
        from models_vit import _create_vision_transformer_Bayes
        # Checkpoints contain all weights, so downloading pretrained weights is unnecessary.
        model = _create_vision_transformer_Bayes(
            'vit_base_patch16_224', pretrained=False, infer=configs.model.infer,
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            num_classes=configs.model.outdim, priorstdout=configs.model.priorstdout,
            mixalpha=configs.train.mixalpha, scalelogstd=configs.model.scalelogstd)
        model.reset_classifier(configs.model.outdim, global_pool=model.global_pool)
        return model
    if name not in ('bmafbresnet18', 'bmafbresnet50', 'resnet18', 'resnet50'):
        raise ValueError('Use the ST evaluation trainer for fully stochastic ResNets')
    from models_deepnet import _setup_model_deep
    return _setup_model_deep(configs)


def make_datasets(configs, root):
    if configs.model.modelname.startswith('vit'):
        if configs.dataset.trainset != 'dtd':
            raise ValueError('ViT evaluation currently supports DTD')
        # Same deterministic preprocessing as the reference ViT OOD evaluator.
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        return {'id': datasets.DTD(str(root), split='test', transform=transform, download=False),
                'svhn2': datasets.SVHN(str(root / 'SVHN'), split='test', transform=transform, download=False),
                'lsun32': ImageDirectory(root / 'LSUN_resize', transform),
                'place365': ImageDirectory(root / 'places365/val_256', transform)}
    from datasets_resnet import _setup_mainset, _setup_oodset
    result = {'id': _setup_mainset(configs)[2].dataset}
    for name in ('svhn2', 'lsun32', 'place365'):
        config = copy.deepcopy(configs)
        config.dataset.oodset = name
        result[name] = _setup_oodset(config).dataset
    return result


def predict(model, dataset, samples, seed, batch, workers, destination, label):
    if destination.exists():
        with np.load(destination) as saved:
            p, y = saved['probabilities'], saved['labels']
        if len(p) != len(dataset):
            raise ValueError(f'Cached dataset size differs: {destination}')
        print(f'{label}: reused {len(p)} saved predictions', flush=True)
        return p, y
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    options = {'prefetch_factor': 1} if workers else {}
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                        pin_memory=True, generator=torch.Generator().manual_seed(seed), **options)
    probabilities, labels = [], []
    start = time.monotonic()
    model.eval()
    with torch.no_grad():
        for index, (images, targets) in enumerate(loader):
            output = model.predict_eval(images.cuda(non_blocking=True), nfsample=samples)
            p = output.pred.detach().cpu().numpy()
            if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
                raise ValueError(f'Invalid probabilities in {label}')
            probabilities.append(p)
            labels.append(targets.numpy())
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(loader):
                print(f'{label}: {min((index+1)*batch,len(dataset))}/{len(dataset)}, '
                      f'{time.monotonic()-start:.1f}s', flush=True)
    p, y = np.concatenate(probabilities), np.concatenate(labels)
    temporary = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, probabilities=p, labels=y)
    temporary.replace(destination)
    return p, y


def entropy(p):
    return -((p + 1e-12) * np.log(p + 1e-12)).sum(axis=1)


def evaluate(path, args):
    print(f'Checkpoint: {path}', flush=True)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    configs = Dict(saved['configs'])
    seed = int(configs.exp.seeds)
    samples = 1 if configs.model.get('infer') == 'map' or configs.model.modelname in ('resnet18','resnet50') else args.nfsamples
    signature = {'checkpoint': str(path.resolve()), 'size': path.stat().st_size,
                 'mtime_ns': path.stat().st_mtime_ns, 'samples': samples, 'seed': seed,
                 'batch': args.batch_size, 'data_root': str(args.data_root.resolve()),
                 'protocol': 'full-test-reference-preprocessing-entropy-ece15-v1'}
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    output = args.output / identity
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps(signature, indent=2)+'\n')
    model = make_model(configs)
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    gc.collect()
    model = model.cuda().eval()
    data = make_datasets(configs, args.data_root)
    counts = {name: len(dataset) for name, dataset in data.items()}
    print(f'Samples={samples}; full dataset counts={counts}', flush=True)
    p, y = predict(model, data.pop('id'), samples, seed, args.batch_size, args.workers, output/'id.npz', 'DTD test' if configs.dataset.trainset=='dtd' else 'ID test')
    id_metrics = {'acc': float((p.argmax(axis=1)==y).mean()), 'nll': float(-get_ll(p,y)),
                  'ece': float(get_ece(p,y,n_bins=15))}
    inside = entropy(p)
    ood = {}
    for name, dataset in data.items():
        probs, _ = predict(model, dataset, samples, seed, args.batch_size, args.workers, output/f'{name}.npz', name)
        outside = entropy(probs)
        ood[name] = float(roc_auc_score(np.r_[np.zeros(len(inside)),np.ones(len(outside))], np.r_[inside,outside]))
    save_seed_results(output/'results.json', configs, path, {str(samples):ood}, list(data),
                      str(args.data_root/'places365/val_256'), id_metrics={str(samples):id_metrics},
                      table_seeds=[seed], table_nfsamples=[samples])
    record=json.loads((output/'results.json').read_text())
    record['evaluation'].update(signature=signature, counts=counts, ece_bins=15, split='test')
    (output/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'FINISHED {path}: {id_metrics}, {ood}; {output}',flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='*',type=Path)
    parser.add_argument('--checkpoint-manifest', type=Path, help='evaluate exactly the checkpoints produced by a training run')
    parser.add_argument('--nfsamples',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--data-root',type=Path,default=Path('/home/yohan/Datasets'))
    parser.add_argument('--output',type=Path,default=Path('results/checkpoint_evaluations'))
    args=parser.parse_args()
    if min(args.nfsamples,args.batch_size)<1 or args.workers<0:
        parser.error('Samples and batch size must be positive; workers must be nonnegative')
    if args.checkpoint_manifest:
        if args.checkpoints:
            parser.error('Use checkpoint paths or --checkpoint-manifest, not both')
        manifest = json.loads(args.checkpoint_manifest.read_text())
        args.checkpoints = [Path(path) for path in manifest['checkpoints']]
    if not args.checkpoints or any(not path.is_file() for path in args.checkpoints):
        parser.error('Provide at least one existing checkpoint')
    torch.set_num_threads(4)
    for path in args.checkpoints:
        evaluate(path,args)
        gc.collect()
        torch.cuda.empty_cache()


if __name__=='__main__':
    main()
