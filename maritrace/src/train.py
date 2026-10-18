"""Train the synthetic five-class segmentation baseline."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("maritrace.train")
DECODER_CHANNELS = (128, 64, 32, 16, 8)


def make_loss(torch, class_weights):
    """Weighted multiclass soft Dice plus focal loss."""
    def loss(logits, target):
        prob = logits.softmax(1)
        onehot = torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2).float()
        dice_per_class = 1 - (2 * (prob * onehot).sum((0, 2, 3)) + 1) / (
            prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1
        )
        dice = (dice_per_class * class_weights).sum() / class_weights.sum()
        pt = prob.gather(1, target[:, None]).squeeze(1).clamp_min(1e-6)
        pixel_weights = class_weights[target]
        focal = ((1 - pt).square() * -pt.log() * pixel_weights).mean()
        return dice + focal
    return loss


def validate_epoch(torch, model, loader, device, loss_fn):
    """Compute validation loss and IoU in one full-resolution inference pass."""
    cm = torch.zeros((5, 5), dtype=torch.int64)
    losses = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)); losses.append(loss_fn(logits, y.to(device)).item())
            pred = logits.argmax(1).cpu()
            idx = y.reshape(-1) * 5 + pred.reshape(-1)
            cm += torch.bincount(idx, minlength=25).reshape(5, 5)
    tp = cm.diag().float()
    union = cm.sum(0) + cm.sum(1) - tp
    return float(np.mean(losses)), (tp / union.clamp_min(1)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--out', default='runs/exp2')
    parser.add_argument('--fast', action='store_true', help='Two-epoch smoke run on 200/40 samples')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-imagenet', action='store_true', help='Skip ImageNet encoder weights')
    args = parser.parse_args()

    import torch
    import segmentation_models_pytorch as smp
    from .dataset import SynthSARDataset
    from .config import CLASS_NAMES

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2 if args.fast else args.epochs
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_ds = SynthSARDataset(args.data, 'train', augment=True, crop_size=128)
    val_ds = SynthSARDataset(args.data, 'val')
    if args.fast:
        train_ds = torch.utils.data.Subset(train_ds, range(min(200, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(40, len(val_ds))))
    freq_ds = SynthSARDataset(args.data, 'train')
    frequencies = freq_ds.class_frequencies()
    # Inverse square root reduces imbalance without allowing rare classes to dominate.
    weights = 1 / np.sqrt(np.maximum(frequencies, 1e-8)); weights = np.minimum(weights / weights.mean(), 8.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    LOGGER.info('device=%s train=%d val=%d crop=128 validation_tile=256', device, len(train_ds), len(val_ds))
    LOGGER.info('class_frequencies=%s', json.dumps(dict(zip(CLASS_NAMES, frequencies.tolist()))))
    LOGGER.info('class_weights=%s', json.dumps(dict(zip(CLASS_NAMES, weights.tolist()))))

    imagenet_loaded = False
    if not args.no_imagenet:
        try:
            model = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', in_channels=1, classes=5, decoder_channels=DECODER_CHANNELS)
            imagenet_loaded = True
            LOGGER.info('ImageNet encoder weights loaded successfully.')
        except Exception as exc:
            LOGGER.warning('ImageNet weights unavailable (%s); initialized encoder randomly.', exc)
            model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=1, classes=5, decoder_channels=DECODER_CHANNELS)
    else:
        model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=1, classes=5, decoder_channels=DECODER_CHANNELS)
        LOGGER.info('ImageNet encoder weights explicitly disabled; initialized encoder randomly.')
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=max(1, min(args.batch_size, 16)), shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = make_loss(torch, class_weights)
    best_oil_iou = -1.0; stale = 0; log_path = out / 'training.csv'

    with log_path.open('w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', *[f'iou_{name}' for name in CLASS_NAMES]])
        for epoch in range(1, epochs + 1):
            model.train(); train_losses = []
            for x, y in train_loader:
                x=x.to(device); y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y); loss.backward(); optimizer.step(); train_losses.append(loss.item())
            model.eval(); val_loss, ious = validate_epoch(torch, model, val_loader, device, loss_fn)
            lr = optimizer.param_groups[0]['lr']
            writer.writerow([epoch, np.mean(train_losses), val_loss, lr, *ious]); fp.flush()
            LOGGER.info('epoch=%d/%d train_loss=%.4f val_loss=%.4f iou=%s', epoch, epochs, np.mean(train_losses), val_loss, json.dumps(dict(zip(CLASS_NAMES, ious))))
            oil_iou = ious[1]
            if oil_iou > best_oil_iou:
                best_oil_iou = oil_iou; stale = 0
                torch.save({'model': model.state_dict(), 'encoder': 'resnet18', 'classes': 5,
                            'imagenet_loaded': imagenet_loaded, 'epoch': epoch,
                            'oil_val_iou': oil_iou, 'class_frequencies': frequencies.tolist(),
                            'class_weights': weights.tolist(), 'seed': args.seed,
                            'input_crop': 128, 'decoder_channels': DECODER_CHANNELS}, out / 'best.pth')
            else:
                stale += 1
            scheduler.step()
            # Preserve the requested minimum-length CPU run before patience can stop it.
            if epoch >= min(30, epochs) and stale >= args.patience:
                LOGGER.info('Early stopping after %d epochs without oil-IoU improvement.', args.patience)
                break
    summary = {'requested_epochs': epochs, 'completed_epochs': epoch, 'device': str(device),
               'best_oil_val_iou': best_oil_iou, 'imagenet_weights_loaded': imagenet_loaded,
               'class_frequencies': dict(zip(CLASS_NAMES, frequencies.tolist())),
               'class_weights': dict(zip(CLASS_NAMES, weights.tolist())), 'seed': args.seed}
    (out / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    LOGGER.info('training complete: %s', json.dumps(summary))


if __name__ == '__main__': main()
