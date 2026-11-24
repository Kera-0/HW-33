import argparse
import json

import torch
from torch import nn

import experiment as e


@torch.inference_mode()
def collect_statistics(model, loader, batches):
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.startswith('bert.encoder.layer')
    }
    statistics = {name: {'sum': None, 'gram': None, 'count': 0} for name in layers}
    hooks = []

    def hook(name):
        def record(module, inputs, output):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            item = statistics[name]
            squared = values.square().sum(dim=0)
            gram = values.T @ values
            item['sum'] = squared if item['sum'] is None else item['sum'] + squared
            item['gram'] = gram if item['gram'] is None else item['gram'] + gram
            item['count'] += values.shape[0]
        return record

    for name, layer in layers.items():
        hooks.append(layer.register_forward_hook(hook(name)))
    model.eval()
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        batch = {k: v.to(e.DEVICE) for k, v in batch.items() if k != 'labels'}
        model(**batch)
    for handle in hooks:
        handle.remove()
    return layers, statistics


def row_mask(scores, sparsity):
    count = int(scores.shape[1] * sparsity)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if count:
        mask.scatter_(1, scores.argsort(dim=1)[:, :count], True)
    return mask


@torch.no_grad()
def wanda(layer, stats, sparsity):
    weight = layer.weight.data
    activation = (stats['sum'] / stats['count']).sqrt()
    weight[row_mask(weight.abs() * activation.unsqueeze(0), sparsity)] = 0


@torch.no_grad()
def sparsegpt(layer, stats, sparsity, block_size=128):
    weight = layer.weight.data.float().clone()
    hessian = stats['gram'] / stats['count']
    diagonal = hessian.diagonal()
    dead = diagonal == 0
    diagonal[dead] = 1
    weight[:, dead] = 0
    diagonal.add_(0.01 * diagonal.mean())
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    inverse = torch.linalg.cholesky(inverse, upper=True)
    for start in range(0, weight.shape[1], block_size):
        end = min(start + block_size, weight.shape[1])
        current = weight[:, start:end].clone()
        block_inverse = inverse[start:end, start:end]
        scores = current.square() / block_inverse.diagonal().square().unsqueeze(0)
        mask = row_mask(scores, sparsity)
        errors = torch.zeros_like(current)
        for column in range(end - start):
            value = current[:, column]
            quantized = value.clone()
            quantized[mask[:, column]] = 0
            error = (value - quantized) / block_inverse[column, column]
            current[:, column:] -= error.unsqueeze(1) * block_inverse[column, column:].unsqueeze(0)
            current[:, column] = quantized
            errors[:, column] = error
        weight[:, start:end] = current
        if end < weight.shape[1]:
            weight[:, end:] -= errors @ inverse[start:end, end:]
    layer.weight.data.copy_(weight.to(layer.weight.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['wanda', 'sparsegpt'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--calibration-batches', type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.sparsity < 1:
        raise ValueError('sparsity must be in [0, 1)')
    e.seed_everything()
    tokenizer, _, tokenized = e.prepare_data()
    data = e.loaders(tokenizer, tokenized, 32)
    model = e.load_state(e.make_student(tokenizer), 'student')
    layers, stats = collect_statistics(model, data['train'], args.calibration_batches)
    for name, layer in layers.items():
        print(args.method, name, flush=True)
        (wanda if args.method == 'wanda' else sparsegpt)(layer, stats[name], args.sparsity)
        del stats[name]
    validation = e.evaluate(model, data['validation'])
    test = e.evaluate(model, data['test'])
    pruned = sum((layer.weight == 0).sum().item() for layer in layers.values())
    total = sum(layer.weight.numel() for layer in layers.values())
    result = {
        'parameters': sum(p.numel() for p in model.parameters()),
        'pruned_linear_fraction': pruned / total,
        'validation_f1': validation['f1'],
        'test_f1': test['f1'],
        'test_report': test['report'],
        'calibration_batches': args.calibration_batches,
    }
    output = e.OUTPUT / args.method
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'best.pt')
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
