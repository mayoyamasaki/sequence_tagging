import nltk
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
devel_sents = list(nltk.corpus.conll2002.iob_sents('esp.testa'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

drop_pos = lambda triplet: (triplet[0], triplet[2])
train_sents = [[drop_pos(t) for t in s] for s in train_sents]
devel_sents = [[drop_pos(t) for t in s] for s in devel_sents]
test_sents = [[drop_pos(t) for t in s] for s in test_sents]

fmt = lambda sents: '\n\n'.join(['\n'.join([' '.join(p) for p in s])
                                 for s in sents])
with open('data/conll2002-train.iob', 'w', encoding='utf-8') as fd:
    fd.write(fmt(train_sents))
with open('data/conll2002-testa.iob', 'w', encoding='utf-8') as fd:
    fd.write(fmt(devel_sents))
with open('data/conll2002-testb.iob', 'w', encoding='utf-8') as fd:
    fd.write(fmt(test_sents))
