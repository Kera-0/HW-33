import json

from torch import nn

import experiment as e
import optional as o
from eval_words import evaluate_words


def main():
    e.seed_everything()
    tokenizer, raw, _ = e.prepare_data()
    stages = [
        'pruned_init', 'sharing', 'sharing_lora', 'intermediate',
        'wanda', 'sparsegpt', 'minilmv2',
    ]
    for stage in stages:
        path = e.OUTPUT / stage / 'best.pt'
        if not path.exists():
            print(stage, 'not ready', flush=True)
            continue
        if stage in ('sharing', 'sharing_lora'):
            model = o.shared_student(tokenizer, stage == 'sharing_lora')
        else:
            model = e.make_student(tokenizer)
        if stage == 'intermediate':
            model.projections = nn.ModuleList(
                [nn.Linear(384, 768).to(e.DEVICE) for _ in range(4)]
            )
        e.load_state(model, stage)
        valid = evaluate_words(model, tokenizer, raw['validation'])
        test = evaluate_words(model, tokenizer, raw['test'])
        result = {
            'validation_f1': valid['f1'],
            'test_f1': test['f1'],
            'test_report': test['report'],
        }
        (e.OUTPUT / stage / 'word_metrics.json').write_text(json.dumps(result, indent=2))
        print(stage, json.dumps(result), flush=True)
        del model


if __name__ == '__main__':
    main()
