import json

import torch
from seqeval.metrics import f1_score, classification_report

import experiment as e


@torch.inference_mode()
def evaluate_words(model, tokenizer, dataset, batch_size=32):
    model.eval()
    targets, predictions = [], []
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        encoded = tokenizer(
            batch['tokens'], is_split_into_words=True, padding=True,
            truncation=True, return_tensors='pt',
        )
        inputs = {key: value.to(e.DEVICE) for key, value in encoded.items()}
        output = model(**inputs).logits.argmax(-1).cpu().tolist()
        for row, labels in enumerate(batch['ner_tags']):
            words = encoded.word_ids(batch_index=row)
            previous, gold, predicted = None, [], []
            for position, word in enumerate(words):
                if word is not None and word != previous:
                    gold.append(e.LABELS[labels[word]])
                    predicted.append(e.LABELS[output[row][position]])
                previous = word
            targets.append(gold)
            predictions.append(predicted)
    return {
        'f1': f1_score(targets, predictions),
        'report': classification_report(targets, predictions, digits=4),
    }


def main():
    e.seed_everything()
    tokenizer, raw, _ = e.prepare_data()
    for stage in ['teacher', 'teacher_retry', 'factorized', 'student']:
        model = e.make_student(tokenizer) if stage == 'student' else e.make_teacher()
        if stage == 'factorized':
            original = model.bert.embeddings.word_embeddings
            model.bert.embeddings.word_embeddings = e.FactorizedEmbedding(
                original.weight, rank=64, padding_idx=tokenizer.pad_token_id
            )
        e.load_state(model, stage)
        valid = evaluate_words(model, tokenizer, raw['validation'])
        test = evaluate_words(model, tokenizer, raw['test'])
        result = {'validation_f1': valid['f1'], 'test_f1': test['f1'], 'test_report': test['report']}
        (e.OUTPUT / stage / 'word_metrics.json').write_text(json.dumps(result, indent=2))
        print(stage, json.dumps(result), flush=True)
        del model


if __name__ == '__main__':
    main()
