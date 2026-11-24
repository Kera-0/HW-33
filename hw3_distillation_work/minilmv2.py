import argparse
import json
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import experiment as e


def capture_relations(model, storage):
    attention = model.bert.encoder.layer[-1].attention.self
    handles = []
    for name in ('query', 'key', 'value'):
        def capture(module, inputs, output, key=name):
            storage[key] = output
        handles.append(getattr(attention, name).register_forward_hook(capture))
    return handles


def relation_loss(student, teacher, attention_mask, heads=12):
    valid = attention_mask.bool()
    total = 0
    for name in ('query', 'key', 'value'):
        relations = []
        for states in (student[name], teacher[name]):
            batch, length, width = states.shape
            vectors = states.float().reshape(batch, length, heads, width // heads).transpose(1, 2)
            scores = vectors @ vectors.transpose(-1, -2) / math.sqrt(width // heads)
            relations.append(scores.masked_fill(~valid[:, None, None, :], -1e4))
        log_prob = F.log_softmax(relations[0], dim=-1)
        target = F.softmax(relations[1], dim=-1)
        per_query = F.kl_div(log_prob, target, reduction='none').sum(-1)
        total += (per_query * valid[:, None, :]).sum() / (valid.sum() * heads)
    return total / 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=20000)
    args = parser.parse_args()
    e.seed_everything()
    tokenizer, _, tokenized = e.prepare_data()
    data = e.loaders(tokenizer, tokenized, args.batch_size)
    teacher = e.load_state(e.make_teacher(), 'teacher')
    student = e.make_student(tokenizer)
    count = sum(p.numel() for p in student.parameters())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    captured_student, captured_teacher = {}, {}
    handles = capture_relations(student, captured_student) + capture_relations(teacher, captured_teacher)
    optimizer = AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)
    steps = min(args.max_steps, args.epochs * len(data['train']))
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * 0.1)), steps)
    folder = e.OUTPUT / 'minilmv2'
    folder.mkdir(parents=True, exist_ok=True)
    best, bad_epochs, step = -1.0, 0, 0
    history = []
    for epoch in range(args.epochs):
        student.train()
        losses = []
        for batch in data['train']:
            batch = {k: v.to(e.DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=e.DEVICE.type == 'cuda'):
                out = student(**batch)
                with torch.no_grad():
                    target = teacher(**batch)
                soft = e.distillation_loss(out.logits, target.logits, batch['labels'])
                relational = relation_loss(captured_student, captured_teacher, batch['attention_mask'])
                loss = out.loss + soft + relational
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            captured_student.clear()
            captured_teacher.clear()
            step += 1
            losses.append(loss.item())
            if step % 100 == 0:
                print(f'minilmv2 step={step}/{steps} loss={loss.item():.4f}', flush=True)
            if step >= steps:
                break
        valid = e.evaluate(student, data['validation'])
        captured_student.clear()
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
    for handle in handles:
        handle.remove()
    e.load_state(student, 'minilmv2')
    test = e.evaluate(student, data['test'])
    result = {
        'parameters': count,
        'best_validation_f1': best,
        'test_f1': test['f1'],
        'test_report': test['report'],
        'steps': step,
    }
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
