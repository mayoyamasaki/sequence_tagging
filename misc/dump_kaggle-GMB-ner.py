# This code dumps ner dataset for this sequence tagger.
# Details are here: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
#
# 1. You download csv file from the project.
#   (1.5) I change the file encode to solve an error on loading binary format csv.
# 2. Run this code with the csv file.
#   ```python3 ./misc/dump_kaggle-GMB-ner.py PATH_TO_CSV```


import argparse
import csv
import logging


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def load_sentences(filename):
    logger = get_logger(__file__)
    with open(filename, 'r', encoding='utf-8') as fd:
        reader = csv.reader(fd)
        _ = next(reader) # pass header

        sentences = []
        sentence = []
        is_dirty = True
        for row in reader:
            try:
                label, token, pos, nertag = row
            except:
                is_dirty = True
                logger.error('Foud iligal row: {}'.format(row))

            # the label of sentence head is like "Sentence N:"
            if label != '':
                if not is_dirty:
                    sentences.append(sentence)
                sentence = []
                is_dirty = False
            sentence.append((token, nertag))

        if len(sentence) > 0:
            sentences.append(sentence)
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Dump kaggle's GMB ner dataset")

    parser.add_argument('csv', type=str, help='csv file')
    args = parser.parse_args()

    sentences = load_sentences(args.csv)
    print('sentence length is {}'.format(len(sentences)))

    train_rate = 0.7
    devel_rate = 0.15
    train_size = int(len(sentences)*train_rate)
    devel_size = int(len(sentences)*devel_rate)
    train_sents = sentences[:train_size]
    devel_sents = sentences[train_size:train_size+devel_size]
    test_sents = sentences[train_size+devel_size:]

    fmt = lambda sents: '\n\n'.join(['\n'.join([' '.join(p) for p in s])
                                     for s in sents])
    with open('data/kaggle-GMB_train.iob', 'w', encoding='utf-8') as fd:
        fd.write(fmt(train_sents[:30]))
    with open('data/kaggle-GMB_devel.iob', 'w', encoding='utf-8') as fd:
        fd.write(fmt(devel_sents[:30]))
    with open('data/kaggle-GMB_test.iob', 'w', encoding='utf-8') as fd:
        fd.write(fmt(test_sents[:30]))

if __name__ == "__main__":
    main()
2017-07-27 22:04:19,961:ERROR: Foud iligal row: ['', ':', 'I-tim']
2017-07-27 22:04:19,981:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:19,981:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:19,995:ERROR: Foud iligal row: ['', 'Kanout薔NNP', 'I-per']
2017-07-27 22:04:20,001:ERROR: Foud iligal row: ['', ',', 'O']
2017-07-27 22:04:20,001:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,002:ERROR: Foud iligal row: ['', ',', 'O']
2017-07-27 22:04:20,002:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,081:ERROR: Foud iligal row: ['', 'Kountch薔NNP', 'I-art']
2017-07-27 22:04:20,087:ERROR: Foud iligal row: ['', 'attach薔NN', 'O']
2017-07-27 22:04:20,226:ERROR: Foud iligal row: ['', 'communiqu薔NN', 'O']
2017-07-27 22:04:20,251:ERROR: Foud iligal row: ['', ':', 'O']
2017-07-27 22:04:20,465:ERROR: Foud iligal row: ['', 'RenNNP', 'I-per']
2017-07-27 22:04:20,468:ERROR: Foud iligal row: ['Sentence: 28979', '``', 'O']
2017-07-27 22:04:20,468:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,472:ERROR: Foud iligal row: ['', 'AndrNNP', 'I-per']
2017-07-27 22:04:20,476:ERROR: Foud iligal row: ['', 'attach薔NNP', 'B-per']
2017-07-27 22:04:20,552:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:20,603:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:20,603:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:20,617:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,617:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,618:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,618:ERROR: Foud iligal row: ['', '``', 'O']
2017-07-27 22:04:20,639:ERROR: Foud iligal row: ['', 'prot馮NN', 'O']
2017-07-27 22:04:20,735:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:20,862:ERROR: Foud iligal row: ['', ';', 'O']
2017-07-27 22:04:20,949:ERROR: Foud iligal row: ['', ';', 'I-org']
