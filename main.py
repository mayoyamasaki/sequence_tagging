import argparse

from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset
from model import NERModel
from config import Config


def train(args):
    # create instance of config
    config = Config(args.config)

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars,
                    lowercase=True, chars=config.chars)
    processing_tag  = get_processing_word(vocab_tags, 
                    lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    dev   = CoNLLDataset(config.dev_filename, processing_word,
                        processing_tag, config.max_iter)
    train = CoNLLDataset(config.train_filename, processing_word,
                        processing_tag, config.max_iter)
    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars))
    model.build()
    model.train(train, dev, vocab_tags)


def evaluate(args):
    # create instance of config
    config = Config(args.config)

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars,
                    lowercase=True, chars=config.chars)
    processing_tag  = get_processing_word(vocab_tags, 
                    lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    test  = CoNLLDataset(config.test_filename, processing_word,
                        processing_tag, config.max_iter)
    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars))
    model.build()
    model.evaluate(test, vocab_tags)


def repl(args):
    # create instance of config
    config = Config(args.config)

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags  = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars,
                    lowercase=True, chars=config.chars)
    processing_tag  = get_processing_word(vocab_tags, 
                    lowercase=False)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # build model
    model = NERModel(config, embeddings, ntags=len(vocab_tags),
                                         nchars=len(vocab_chars))
    model.build()
    model.interactive_shell(vocab_tags, processing_word)


def main():
    parser = argparse.ArgumentParser(description='Sequence Tagger')
    subparsers = parser.add_subparsers()

    parser_add = subparsers.add_parser('train', help='see `train -h`')
    parser_add.add_argument('config', type=str, help='config.ini file')
    parser_add.set_defaults(handler=train)

    parser_add = subparsers.add_parser('evaluate', help='see `train -h`')
    parser_add.add_argument('config', type=str, help='config.ini file')
    parser_add.set_defaults(handler=evaluate)

    parser_add = subparsers.add_parser('repl', help='see `train -h`')
    parser_add.add_argument('config', type=str, help='config.ini file')
    parser_add.set_defaults(handler=repl)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
