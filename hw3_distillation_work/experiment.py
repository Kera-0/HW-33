import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/ashaba1in-conll2003'
BASE = ROOT / 'data/bert-base-cased'
OUTPUT = ROOT / 'hw3_distillation_work/results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def align_labels_with_tokens(labels, tokenized):
    result = []
    previous_word = None
    word_ids = tokenized.word_ids
    for word in word_ids() if callable(word_ids) else word_ids:
        if word is None:
            result.append(-100)
        else:
            label = labels[word]
            if word == previous_word and label > 0 and label % 2 == 1:
                label += 1
            result.append(label)
        previous_word = word
    return result


def prepare_data():
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    files = {
        split: str(DATA / f'{split}-00000-of-00001.parquet')
        for split in ('train', 'validation', 'test')
    }
    raw = load_dataset('parquet', data_files=files)

    def tokenize(batch):
        encoded = tokenizer(
            batch['tokens'], is_split_into_words=True, truncation=True, max_length=512
        )
        encoded['labels'] = [
            align_labels_with_tokens(tags, encoded[i])
            for i, tags in enumerate(batch['ner_tags'])
        ]
        return encoded

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw['train'].column_names)
    return tokenizer, raw, tokenized


def loaders(tokenizer, tokenized, batch_size):
    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if DEVICE.type == 'cuda' else None
    )
    return {
        name: DataLoader(
            part,
            batch_size=batch_size,
            shuffle=name == 'train',
            collate_fn=collator,
        )
        for name, part in tokenized.items()
    }


def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0):
    mask = labels != -100
    student_log_probs = F.log_softmax(student_logits[mask].float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits[mask].float() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * temperature**2


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    targets, predictions = [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        predicted = logits.argmax(dim=-1).cpu().tolist()
        labels = batch['labels'].cpu().tolist()
        for p, y in zip(predicted, labels):
            targets.append([LABELS[t] for t in y if t != -100])
            predictions.append([LABELS[a] for a, t in zip(p, y) if t != -100])
    return {
        'f1': f1_score(targets, predictions),
        'report': classification_report(targets, predictions, digits=4),
    }


class FactorizedEmbedding(nn.Module):
    def __init__(self, matrix, rank=64, padding_idx=0):
        super().__init__()
        with torch.no_grad():
            u, s, v = torch.svd_lowrank(matrix.detach().float(), q=rank + 16, niter=4)
            s = s[:rank].sqrt()
            left = u[:, :rank] * s
            right = s[:, None] * v[:, :rank].T
        self.embedding = nn.Embedding.from_pretrained(left, freeze=False, padding_idx=padding_idx)
        self.projection = nn.Linear(rank, matrix.shape[1], bias=False, device=matrix.device)
        self.projection.weight.data.copy_(right.T)

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))


def make_teacher():
    return AutoModelForTokenClassification.from_pretrained(
        BASE, num_labels=len(LABELS), local_files_only=True
    ).to(DEVICE)


def make_student(tokenizer):
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=1536,
        num_labels=len(LABELS),
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForTokenClassification(config).to(DEVICE)
    assert sum(p.numel() for p in model.parameters()) <= 20_000_000
    return model


def load_state(model, stage):
    path = OUTPUT / stage / 'best.pt'
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model


def train(stage, model, data, epochs, batch_size, lr, max_steps, patience, teacher=None):
    folder = OUTPUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    count = sum(p.numel() for p in model.parameters())
    print(f'{stage}: {count:,} parameters on {DEVICE}', flush=True)
    train_loader = data['train']
    steps = min(max_steps, epochs * len(train_loader)) if max_steps else epochs * len(train_loader)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(steps * 0.1)),
        num_training_steps=steps,
    )
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    best, bad_epochs, step = -1.0, 0, 0
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_steps = 0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
                out = model(**batch)
                loss = out.loss
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(**batch).logits
                    loss = loss + distillation_loss(out.logits, teacher_logits, batch['labels'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            step += 1
            epoch_steps += 1
            if step % 100 == 0:
                print(f'{stage} step={step}/{steps} loss={loss.item():.4f}', flush=True)
            if step >= steps:
                break
        valid = evaluate(model, data['validation'])
        record = {
            'epoch': epoch + 1,
            'step': step,
            'train_loss': total_loss / epoch_steps,
            'validation_f1': valid['f1'],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        (folder / 'history.json').write_text(json.dumps(history, indent=2))
        if valid['f1'] > best:
            best, bad_epochs = valid['f1'], 0
            torch.save(model.state_dict(), folder / 'best.pt')
        else:
            bad_epochs += 1
        if step >= steps or bad_epochs >= patience:
            break
    load_state(model, stage)
    test = evaluate(model, data['test'])
    summary = {
        'parameters': count,
        'best_validation_f1': best,
        'test_f1': test['f1'],
        'test_report': test['report'],
        'steps': step,
    }
    (folder / 'metrics.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['teacher', 'factorized', 'student'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=0)
    args = parser.parse_args()
    seed_everything()
    tokenizer, raw, tokenized = prepare_data()
    print('Dataset:', {k: len(v) for k, v in raw.items()}, flush=True)
    data = loaders(tokenizer, tokenized, args.batch_size)
    if args.stage == 'teacher':
        train('teacher', make_teacher(), data, args.epochs or 5, args.batch_size, 3e-5, args.max_steps, 3)
    elif args.stage == 'factorized':
        model = load_state(make_teacher(), 'teacher')
        original = model.bert.embeddings.word_embeddings
        model.bert.embeddings.word_embeddings = FactorizedEmbedding(
            original.weight, rank=64, padding_idx=tokenizer.pad_token_id
        )
        train('factorized', model, data, args.epochs or 5, args.batch_size, 2e-5, args.max_steps, 3)
    else:
        teacher = load_state(make_teacher(), 'teacher')
        train(
            'student', make_student(tokenizer), data, args.epochs or 100,
            args.batch_size, 3e-4, args.max_steps or 20_000, 8, teacher=teacher,
        )


if __name__ == '__main__':
    main()
