# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from logging import getLogger
from collections import defaultdict



logger = getLogger()


BOS_WORD = '[CLS]'
EOS_WORD = '[SEP]'
PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'

SPECIAL_WORD = '<special%i>'
SPECIAL_WORDS = 10

SEP_WORD = '[SEP]'
MASK_WORD = '[MASK]'


class Dictionary(object):

    def __init__(self, id2word, word2id, counts):
        assert len(id2word) == len(word2id) == len(counts)
        self.id2word = id2word
        self.word2id = word2id
        self.counts = counts
        self.bos_index = word2id[BOS_WORD]
        self.eos_index = word2id[EOS_WORD]
        self.pad_index = word2id[PAD_WORD]
        self.unk_index = word2id[UNK_WORD]
        self.check_valid()

    def __len__(self):
        """
        Returns the number of words in the dictionary.
        """
        return len(self.id2word)

    def __getitem__(self, i):
        """
        Returns the word of the specified index.
        """
        return self.id2word[i]

    def __contains__(self, w):
        """
        Returns whether a word is in the dictionary.
        """
        return w in self.word2id

    def __eq__(self, y):
        """
        Compare this dictionary with another one.
        """
        self.check_valid()
        y.check_valid()
        if len(self.id2word) != len(y):
            return False
        return all(self.id2word[i] == y[i] for i in range(len(y)))

    def check_valid(self):
        """
        Check that the dictionary is valid.
        """
        assert self.bos_index == 2
        assert self.eos_index == 3
        assert self.pad_index == 0
        assert self.unk_index == 1
        assert len(self.id2word) == len(self.word2id) == len(self.counts)
        assert set(self.word2id.keys()) == set(self.counts.keys())
        for i in range(len(self.id2word)):
            assert self.word2id[self.id2word[i]] == i
        

    def index(self, word, no_unk=False):
        """
        Returns the index of the specified word.
        """
        if no_unk:
            return self.word2id[word]
        else:
            return self.word2id.get(word, self.unk_index)

    def max_vocab(self, max_vocab):
        """
        Limit the vocabulary size.
        """
        return
        assert max_vocab >= 1
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if k < max_vocab}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info("Maximum vocabulary size: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (max_vocab, init_size, len(self), init_size - len(self)))

    def min_count(self, min_count):
        """
        Threshold on the word frequency counts.
        """
        return
        assert min_count >= 0
        init_size = len(self)
        self.id2word = {k: v for k, v in self.id2word.items() if self.counts[self.id2word[k]] >= min_count or k < 4 + SPECIAL_WORDS}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.counts = {k: v for k, v in self.counts.items() if k in self.word2id}
        self.check_valid()
        logger.info("Minimum frequency count: %i. Dictionary size: %i -> %i (removed %i words)."
                    % (min_count, init_size, len(self), init_size - len(self)))

    @staticmethod
    def read_vocab(vocab_path):
        """
        Create a dictionary from a vocabulary file.
        """
        skipped = 0
        assert os.path.isfile(vocab_path), vocab_path
        word2id = {PAD_WORD: 0, UNK_WORD: 1, BOS_WORD: 2, EOS_WORD: 3, MASK_WORD: 4}
        counts = {k: 0 for k in word2id.keys()}
        f = open(vocab_path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if '\u2028' in line:
                skipped += 1
                continue
            line = line.rstrip().split()
            if len(line) != 2:
                skipped += 1
                continue
            assert len(line) == 2, (i, line)
            # assert line[0] not in word2id and line[1].isdigit(), (i, line)
            assert line[1].isdigit(), (i, line)
            if line[0] in word2id:
                skipped += 1
                print('%s already in vocab' % line[0])
                continue
            if not line[1].isdigit():
                skipped += 1
                print('Empty word at line %s with count %s' % (i, line))
                continue
            word2id[line[0]] = 5 + i - skipped  # shift because of extra words
            counts[line[0]] = int(line[1])
        f.close()
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, counts)
        logger.info("Read %i words from the vocabulary file." % len(dico))
        if skipped > 0:
            logger.warning("Skipped %i empty lines!" % skipped)
        return dico

    @staticmethod
    def load_weights(path):
        

        if not os.path.isfile(path + '.weights'):
            return None
        print("Loading weights")
        weights = []

        f = open(path + '.weights', 'r', encoding='utf-8')
        for i, line in enumerate(f):
            sentence_weights = [int(w) for w in list(line.rstrip())]
            weights.extend(sentence_weights)
            weights.append(1)
            if i % 1000000 == 0 and i > 0:
                print(i)
        
        return np.int8(weights)

    @staticmethod
    def index_data(path, bin_path, dico):
        """
        Index sentences with a dictionary.
        """
        # if bin_path is not None and os.path.isfile(bin_path):
        #     print("Loading data from %s ..." % bin_path)
        #     data = torch.load(bin_path)
        #     assert dico == data['dico']
        #     return data

        positions = []
        sentences = []
        unk_words = {}
        weights = Dictionary.load_weights(path)

        # index sentences
        f = open(path, 'r', encoding='utf-8')
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(i)
            s = list(line.rstrip())
            # skip empty sentences
            if len(s) == 0:
                print("Empty sentence in line %i." % i)
            # index sentence words
            count_unk = 0
            indexed = []
            for w in s:
                word_id = dico.index(w, no_unk=False)

                assert word_id >= 0
                indexed.append(word_id)
                if word_id == dico.unk_index:
                    unk_words[w] = unk_words.get(w, 0) + 1
                    count_unk += 1
            # add sentence
            positions.append([len(sentences), len(sentences) + len(indexed)])
            sentences.extend(indexed)
            sentences.append(dico.eos_index)  # EOS index
        f.close()


        # tensorize data
        positions = np.int64(positions)
        if len(dico) < 1 << 8:
            sentences = np.uint8(sentences)
        elif len(dico) < 1 << 16:
            sentences = np.uint16(sentences)
        elif len(dico) < 1 << 31:
            sentences = np.int32(sentences)
        else:
            raise Exception("Dictionary is too big.")

        assert sentences.min() >= 0

        data = {
            'dico': dico,
            'positions': positions,
            'sentences': sentences,
            'unk_words': unk_words,
            'weights': weights,
        }

        if bin_path is not None:
            print("Saving the data to %s ..." % bin_path)
            torch.save(data, bin_path, pickle_protocol=4)

        return data