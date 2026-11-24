import argparse
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import experiment as e


class LayerWithAdapter(nn.Module):
    def __init__(self, layer, hidden_size=384, rank=16):
        super().__init__()
        self.layer = layer
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, *args, **kwargs):
        inputs = args[0] if args else kwargs['hidden_states']
        result = self.layer(*args, **kwargs)
        hidden = result[0]
        return (hidden + self.up(self.down(inputs)), *result[1:])


def shared_student(tokenizer, adapters=False):
    student = e.make_student(tokenizer)
    shared = student.bert.encoder.layer[0]
    if adapters:
        layers = [LayerWithAdapter(shared) for _ in range(4)]
    else:
        layers = [shared for _ in range(4)]
    student.bert.encoder.layer = nn.ModuleList(layers).to(e.DEVICE)
    return student


def pruned_student(tokenizer, teacher):
    student = e.make_student(tokenizer)
    source = teacher.state_dict()
    selected = {}
    for name, target in student.state_dict().items():
        parts = name.split('.')
        if len(parts) > 3 and parts[:3] == ['bert', 'encoder', 'layer']:
            parts[3] = str((int(parts[3]) + 1) * 3 - 1)
        original = source['.'.join(parts)]
        slices = tuple(slice(0, size) for size in target.shape)
        value = original[slices]
        if value.shape != target.shape:
            raise ValueError(f'Cannot prune {name}: {original.shape} -> {target.shape}')
        selected[name] = value.clone()
    student.load_state_dict(selected)
    return student


def train_intermediate(student, teacher, data, batch_size, epochs, max_steps):
    stage = 'intermediate'
    folder = e.OUTPUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    student.projections = nn.ModuleList([nn.Linear(384, 768).to(e.DEVICE) for _ in range(4)])
    count = sum(p.numel() for p in student.parameters())
    assert count <= 20_000_000
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    steps = min(max_steps, epochs * len(data['train']))
    optimizer = AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * 0.1)), steps)
    best, bad_epochs, step = -1.0, 0, 0
    history = []
    print(f'{stage}: {count:,} parameters on {e.DEVICE}', flush=True)
    for epoch in range(epochs):
        student.train()
        losses = []
        for batch in data['train']:
            batch = {k: v.to(e.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=e.DEVICE.type == 'cuda'):
                out = student(**batch, output_hidden_states=True)
                with torch.no_grad():
                    target = teacher(**batch, output_hidden_states=True)
                mask = batch['labels'].ne(-100)
                soft = e.distillation_loss(out.logits, target.logits, batch['labels'])
                hidden = sum(
                    F.mse_loss(
                        student.projections[i](out.hidden_states[i + 1])[mask].float(),
                        target.hidden_states[(i + 1) * 3][mask].float(),
                    )
                    for i in range(4)
                ) / 4
                loss = out.loss + soft + hidden
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f'{stage} step={step}/{steps} loss={loss.item():.4f}', flush=True)
            if step >= steps:
                break
        valid = e.evaluate(student, data['validation'])
        history.append({
            'epoch': epoch + 1,
            'step': step,
            'train_loss': sum(losses) / len(losses),
            'validation_f1': valid['f1'],
        })
        print(json.dumps(history[-1]), flush=True)
        (folder / 'history.json').write_text(json.dumps(history, indent=2))
        if valid['f1'] > best:
            best, bad_epochs = valid['f1'], 0
            torch.save(student.state_dict(), folder / 'best.pt')
        else:
            bad_epochs += 1
        if step >= steps or bad_epochs >= 8:
            break
    e.load_state(student, stage)
    test = e.evaluate(student, data['test'])
    summary = {
        'parameters': count,
        'best_validation_f1': best,
        'test_f1': test['f1'],
        'test_report': test['report'],
        'steps': step,
    }
    (folder / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['sharing', 'sharing_lora', 'intermediate', 'pruned_init'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=20000)
    args = parser.parse_args()
    e.seed_everything()
    tokenizer, _, tokenized = e.prepare_data()
    data = e.loaders(tokenizer, tokenized, args.batch_size)
    teacher = e.load_state(e.make_teacher(), 'teacher')
    if args.stage == 'intermediate':
        train_intermediate(e.make_student(tokenizer), teacher, data, args.batch_size, args.epochs, args.max_steps)
        return
    if args.stage == 'pruned_init':
        student = pruned_student(tokenizer, teacher)
    else:
        student = shared_student(tokenizer, args.stage == 'sharing_lora')
    e.train(args.stage, student, data, args.epochs, args.batch_size, 3e-4, args.max_steps, 8, teacher=teacher)


if __name__ == '__main__':
    main()
