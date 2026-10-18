"""Evaluate saved segmentation checkpoints on held-out synthetic tiles."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ('sea', 'oil', 'lookalike', 'ship', 'land')


def metrics(truth, pred, classes=5):
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(truth).ravel(), np.asarray(pred).ravel()), 1)
    return metrics_from_confusion(cm)


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64); tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(0), 1); recall = tp / np.maximum(cm.sum(1), 1)
    union = cm.sum(0) + cm.sum(1) - tp
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {'confusion_matrix': cm.tolist(), 'iou': (tp / np.maximum(union, 1)).tolist(),
            'f1': f1.tolist(), 'precision': precision.tolist(), 'recall': recall.tolist(),
            'pixel_accuracy': float(tp.sum() / max(cm.sum(), 1))}


def evaluate_checkpoint(model_path, data_path, out_path, compare_path=None):
    import torch
    import segmentation_models_pytorch as smp
    from PIL import Image
    from .dataset import SynthSARDataset

    torch.set_num_threads(min(4, torch.get_num_threads())); torch.manual_seed(1701)
    out = Path(out_path); out.mkdir(parents=True, exist_ok=True)
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    model = smp.Unet(encoder_name=state.get('encoder', 'resnet18'), encoder_weights=None, in_channels=1, classes=5,
                     decoder_channels=state.get('decoder_channels', (256, 128, 64, 32, 16)))
    model.load_state_dict(state['model']); model.eval()
    ds = SynthSARDataset(data_path, 'test'); cm = np.zeros((5, 5), dtype=np.int64); robust_cm = np.zeros_like(cm)
    rng = torch.Generator().manual_seed(1701); color = np.array([[20,50,75],[230,40,20],[220,180,40],[245,245,235],[90,145,80]], dtype=np.uint8)
    preview = []
    with torch.inference_mode():
        for start in range(0, len(ds), 8):
            samples = [ds[i] for i in range(start, min(start + 8, len(ds)))]
            x = torch.stack([a for a, _ in samples]); y = torch.stack([b for _, b in samples])
            pred = model(x).argmax(1)
            # Stress both contrast and speckle/noise in one deterministic perturbation.
            perturbed = (x * .65 + torch.randn(x.shape, generator=rng) * .08).clamp(0, 1)
            robust_pred = model(perturbed).argmax(1)
            ids = (y * 5 + pred).reshape(-1).numpy(); cm += np.bincount(ids, minlength=25).reshape(5, 5)
            ids = (y * 5 + robust_pred).reshape(-1).numpy(); robust_cm += np.bincount(ids, minlength=25).reshape(5, 5)
            for j in range(min(12 - len(preview), len(samples))):
                base = (x[j, 0].numpy() * 255).astype('uint8'); gt = y[j].numpy().astype('uint8'); pr = pred[j].numpy()
                overlay = np.stack([base] * 3, -1); overlay[pr == 1] = [255, 30, 20]
                preview.extend([Image.fromarray(base).convert('RGB'), Image.fromarray(color[gt]), Image.fromarray(color[pr]), Image.fromarray(overlay)])
    result = metrics_from_confusion(cm); robust = metrics_from_confusion(robust_cm)
    result['robustness_noisy_lower_contrast_pixel_accuracy'] = robust['pixel_accuracy']
    result['robustness_pixel_accuracy_drop'] = result['pixel_accuracy'] - robust['pixel_accuracy']
    result['robustness_iou'] = robust['iou']; result['data_notice'] = 'Synthetic test split only; real SAR performance is not demonstrated.'
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    if compare_path:
        old = json.loads(Path(compare_path).read_text()); rows=[]
        for i, name in enumerate(CLASS_NAMES):
            row={'class':name}
            for run, met in (('exp1',old),('exp2',result)):
                for key in ('iou','f1','precision','recall'): row[f'{run}_{key}']=met[key][i]
            rows.append(row)
        with (out/'comparison.csv').open('w',newline='') as fp:
            writer=csv.DictWriter(fp,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (out/'comparison.json').write_text(json.dumps(rows,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set_xticks(range(5),CLASS_NAMES,rotation=30); ax.set_yticks(range(5),CLASS_NAMES); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png'); plt.close(fig)
    rows = list(csv.DictReader((out/'training.csv').open(newline=''))) if (out/'training.csv').exists() else []
    if rows:
        fig, ax=plt.subplots(); ax.plot([int(r['epoch']) for r in rows],[float(r['train_loss']) for r in rows],label='train')
        ax.plot([int(r['epoch']) for r in rows],[float(r['val_loss']) for r in rows],label='validation'); ax.legend(); ax.set_xlabel('epoch'); ax.set_ylabel('loss'); fig.savefig(out/'training_curves.png'); plt.close(fig)
    grid=Image.new('RGB',(1024,768),'white')
    for i,tile in enumerate(preview): grid.paste(tile.resize((256,256)),((i%4)*256,(i//4)*256))
    grid.save(out/'prediction_grid.png')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--data',default='data/synthetic'); parser.add_argument('--model',default='runs/exp2/best.pth'); parser.add_argument('--out',default='runs/exp2'); parser.add_argument('--compare')
    args=parser.parse_args(); evaluate_checkpoint(args.model,args.data,args.out,args.compare)
