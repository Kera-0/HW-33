import experiment as e

e.seed_everything(123)
tokenizer, _, tokenized = e.prepare_data()
data = e.loaders(tokenizer, tokenized, 32)
e.train('teacher_retry', e.make_teacher(), data, 6, 32, 2e-5, 0, 3)
