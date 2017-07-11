import nltk
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

drop_pos = lambda triplet: (triplet[0], triplet[2])
train_sents = [[drop_pos(t) for t in s] for s in train_sents]
test_sents = [[drop_pos(t) for t in s] for s in test_sents]

i = int(len(train_sents) * 0.8)
train_sents, devel_sents = train_sents[:i], train_sents[i:]

fmt = lambda sents: '\n\n'.join(['\n'.join([' '.join(p) for p in s])
                                 for s in sents])
with open('data/train.iob', 'w', encoding='utf-8') as fd:
    fd.write(fmt(train_sents))
with open('data/devel.iob', 'w', encoding='utf-8') as fd:
    fd.write(fmt(devel_sents))
with open('data/test.iob', 'w', encoding='utf-8') as fd:
    fd.write(fmt(test_sents))
