"""Independent, label-free pretrained inference and retrieval evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data import load_data
from model import PCH
from metrics import rank_and_evaluate


def load_model(checkpoint_dir, device):
    folder = Path(checkpoint_dir)
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    checkpoint = folder / 'model.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != config['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 does not match config.json.')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = PCH(SimpleNamespace(**config))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    return model.to(device).eval(), config


@torch.inference_mode()
def encode(model, image, text, batch_size, device):
    codes = []
    for start in range(0, len(image), batch_size):
        i = torch.from_numpy(image[start:start + batch_size]).to(device)
        t = torch.from_numpy(text[start:start + batch_size]).to(device)
        continuous = model(i, t)['code'].cpu().numpy()
        if not np.isfinite(continuous).all():
            raise ValueError('Model produced non-finite codes.')
        codes.append(np.where(continuous >= 0, 1, -1).astype(np.int8))
    return np.concatenate(codes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Directory with model.pt and config.json')
    parser.add_argument('--data_path', help='D:\code\datasets\clip_dataset(caption+index+label)\flickr25k')
    parser.add_argument('--data_root', help='Root with flickr25k/, coco/, nuswide/')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='evaluation_results')
    args = parser.parse_args()
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model, config = load_model(args.checkpoint, device)
    dset, _ = load_data(config['dataset'], args.data_root, args.data_path)
    for prefix, arrays in [('database', (dset.I_db, dset.T_db, dset.L_db)), ('query', (dset.I_te, dset.T_te, dset.L_te))]:
        for kind, array, width in zip(['image', 'text', 'label'], arrays, [config['image_dim'], config['text_dim'], config['classes']]):
            if not len(array) or array.shape[1] != width or not np.isfinite(array).all():
                raise ValueError(f'Invalid {prefix} {kind} data: {array.shape}')
            expected = config['expected_data_shapes'][f'{prefix}_{kind}']
            if list(array.shape) != expected:
                print(f'Notice: {prefix} {kind} shape differs from reference {expected}.')
    print('Encoding database...', flush=True)
    database = encode(model, dset.I_db, dset.T_db, config['eval_batch_size'], device)
    print('Encoding queries...', flush=True)
    query = encode(model, dset.I_te, dset.T_te, config['eval_batch_size'], device)
    print('Computing retrieval metrics...', flush=True)
    metrics, topk, precision, recall, pr, timings = rank_and_evaluate(query, database, dset.L_te, dset.L_db, SimpleNamespace(**config))
    expected = json.loads((Path(args.checkpoint) / 'expected_metrics.json').read_text(encoding='utf-8'))
    metrics['difference_from_recorded'] = {k: metrics[k] - expected[k] for k in ['MAP_all', 'MAP_at_50']}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps({**metrics, **timings}, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'hashcodes_and_labels.npz', query_codes=query, database_codes=database, query_labels=dset.L_te, database_labels=dset.L_db)
    for name, header, rows in [('precision_topk.csv', ['K', 'precision'], zip(topk, precision)), ('pr_curve.csv', ['recall', 'precision'], zip(recall, pr))]:
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
