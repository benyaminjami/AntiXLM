# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import Bio
import subprocess
from Bio.Align import substitution_matrices
from collections import OrderedDict
import numpy as np
import torch
from ..model.transformer import get_masks
from ..utils import to_cuda, restore_segmentation, concat_batches
# from ..model.memory import HashingMemory


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)


logger = getLogger()


def kl_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    _x = x.copy()
    _x[x == 0] = 1
    return np.log(len(x)) + (x * np.log(_x)).sum()


def gini_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    B = np.cumsum(np.sort(x)).mean()
    return 1 - 2 * B


def tops(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    y = np.cumsum(np.sort(x))
    top50, top90, top99 = y.shape[0] - np.searchsorted(y, [0.5, 0.1, 0.01])
    return top50, top90, top99


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params
        

        # create directory to store hypotheses, and reference files for BLEU evaluation
        params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path
        if self.params.is_master:
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def get_iterator(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        if len(self.params.langs) > 30:
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
            subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
            n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
        elif len(self.params.langs) > 5:
            subsample = 10 if data_set == 'test' else 5
            n_sentences = 300 if data_set == 'test' else 1500
        else:
            # n_sentences = -1 if data_set == 'valid' else 100
            n_sentences = -1
            subsample = 1

        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                iterator = self.data['mono'][lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1, w1), (sent2, len2, w2) in self.get_iterator(data_set, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})
        data_set_dictionary = {
            "ag-ab.test":0,
            "ab-ag.test":1,
            "ag-ab.valid":2,
            "ab-ag.valid":3,
            "ag-ab.valid-cdr":4, 
            "ag-ab.valid-cdr_open":5, 
            "ag-ab.test-cdr":6, 
            "ag-ab.test-cdr_open":7 
        }
        try:
            os.mkdir(os.path.join(params.hyp_path, str(scores['epoch'])))
        except:
            print('Folder exists')
        with torch.no_grad():

            for data_set in ['valid', 'test']:

                for lang in params.ae_steps:
                    self.evaluate_mt(scores, data_set, lang, lang, False)

                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):                    
                    eval_bleu = params.eval_bleu and \
                        data_set_dictionary["{0}-{1}.{2}".format(lang1, lang2, data_set)] % (params.world_size) == params.node_id
                    if eval_bleu:
                        self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):                    
                    eval_bleu = params.eval_bleu and \
                        data_set_dictionary["{0}-{1}.{2}-cdr".format(lang1, lang2, data_set)] % (params.world_size) == params.node_id
                    if eval_bleu:
                        self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu, cdr=True)
                
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):                    
                    eval_bleu = params.eval_bleu and params.open and \
                        data_set_dictionary["{0}-{1}.{2}-cdr_open".format(lang1, lang2, data_set)] % (params.world_size) == params.node_id
                    if eval_bleu:
                        self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)
                    

        return scores

class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params, bert):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.cuda = params.cuda
        self.params = params
        self.fused_bert = params.fused_bert
        if self.fused_bert:
            self.bert = bert
        self.aligner = Bio.Align.PairwiseAligner()
        self.aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")


    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs


        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_sentences = 0
        n_words = 0
        xe_loss = 0
        n_valid = 0
        generate_identity, generate_cdr_identity = 0, 0
        cdr_generate_identity, cdr_generate_cdr_identity = 0, 0
        open_cdr_generate_identity, open_cdr_generate_cdr_identity = 0, 0
        forward_identity, forward_cdr_identity = 0, 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []
            forward_result = []
            cdr_hypothesis = []
            cdr_open_hypothesis = []

        for batch in self.get_iterator(data_set, lang1, params.id2lang[int(not lang1_id)]):

            # generate batch
            (x1, len1, w1), (x2, len2, w2) = batch

            if lang1 == lang2:
                x2, len2 = x1, len1
            
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (alen[:, None] < len2[None] - 1)[:-1]   # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask)
            
            token_type_ids = torch.zeros_like(x1)
            # assert len(y) == (len2 - 1).sum().item()
            
            # cuda
            # TODO: GPU
            if self.params.cuda:
                x1, len1, langs1, x2, len2, langs2, y, token_type_ids = to_cuda(x1, len1, langs1, x2, len2, langs2, y, token_type_ids)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.params.amp == 1)):
                bert_embed = None
                if self.fused_bert:    
                    bert_embed = self.bert(input_ids=x1.T, token_type_ids=token_type_ids.T, attention_mask=get_masks(x1.size()[0], len1, False)[0]).last_hidden_state
                # encode source sentence
                enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, bert_embed=bert_embed)
                enc1 = enc1.transpose(0, 1)

                # decode target sentence
                dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1, bert_embed=bert_embed)

                # loss
                word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

                # update stats
                n_sentences += x1.shape[1]
                n_words += y.size(0)
                xe_loss += loss.item() * len(y)
                n_valid += (word_scores.max(1)[1] == y).sum().item()
                
                if eval_bleu:
                    ref = convert_to_text(x2, len2, self.dico, params)

                    if params.beam_size == 1:
                        generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=self.params.max_len[lang2], bert_embed=bert_embed)
                    else:
                        if lang2 == 'ab':
                            cdr_generated, cdr_lengths = decoder.generate_beam(
                                enc1, len1, lang2_id, beam_size=params.beam_size,
                                length_penalty=params.length_penalty,
                                early_stopping=params.early_stopping,
                                max_len=self.params.max_len[lang2] + 2,
                                bert_embed=bert_embed,
                                antibody=True,
                                tgt_frw=x2,
                                w=w2
                            )
                            cdr_hypothesis.extend(convert_to_text(cdr_generated, cdr_lengths, self.dico, params))
                            batch_generate_identity, batch_generate_cdr_identity = self.calculate_identity(ref, cdr_hypothesis, lang2, w2, beam_size=params.beam_size)
                            cdr_generate_identity += batch_generate_identity
                            cdr_generate_cdr_identity += batch_generate_cdr_identity

                            cdr_open_generated, cdr_open_lengths = decoder.generate_beam(
                                enc1, len1, lang2_id, beam_size=params.beam_size,
                                length_penalty=params.length_penalty,
                                early_stopping=params.early_stopping,
                                max_len=self.params.max_len[lang2] + 2,
                                bert_embed=bert_embed,
                                antibody=True,
                                tgt_frw=x2,
                                w=w2,
                                open_end=True
                            )
                            cdr_open_hypothesis.extend(convert_to_text(cdr_open_generated, cdr_open_lengths, self.dico, params))
                            batch_generate_identity, batch_generate_cdr_identity = self.calculate_identity(ref, cdr_open_hypothesis, lang2, w2, beam_size=params.beam_size)
                            open_cdr_generate_identity += batch_generate_identity
                            open_cdr_generate_cdr_identity += batch_generate_cdr_identity
                            

                        generated, lengths = decoder.generate_beam(
                            enc1, len1, lang2_id, beam_size=params.beam_size,
                            length_penalty=params.length_penalty,
                            early_stopping=params.early_stopping,
                            max_len=self.params.max_len[lang2] + 2,
                            bert_embed=bert_embed,
                            antibody=False,
                        )
                        
                    
                    hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))
                    forward_result.extend(convert_to_text(word_scores.max(1)[1],
                        len2, 
                        self.dico, 
                        params, 
                        'ae',
                        torch.arange(len(len2), dtype=torch.long, device=pred_mask.device).repeat(pred_mask.shape[0],1).masked_select(pred_mask)))
                    
                    
                    
                    batch_forward_identity, batch_forward_cdr_identity = self.calculate_identity(ref, forward_result, lang2, w2)
                    batch_generate_identity, batch_generate_cdr_identity = self.calculate_identity(ref, hypothesis, lang2, w2, beam_size=params.beam_size)

                    forward_identity += batch_forward_identity
                    forward_cdr_identity += batch_forward_cdr_identity

                    generate_identity += batch_generate_identity
                    generate_cdr_identity += batch_generate_cdr_identity


        if lang1 != lang2:
            # compute perplexity and prediction accuracy
            scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
            scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words
            if eval_bleu:
                scores['%s-%s-%s_mt_frw_identity' % (data_set, lang1, lang2)] = float(100. * forward_identity / n_sentences)
                scores['%s-%s-%s_mt_gen_identity' % (data_set, lang1, lang2)] = float(100. * generate_identity / n_sentences)
                if lang2 == 'ab':
                    scores['%s-%s-%s_mt_frw_cdr_identity' % (data_set, lang1, lang2)] = float(100. * forward_cdr_identity / n_sentences)
                    scores['%s-%s-%s_mt_gen_cdr_identity' % (data_set, lang1, lang2)] = float(100. * generate_cdr_identity / n_sentences)
                    scores['%s-%s-%s_mt_cdr_gen_identity' % (data_set, lang1, lang2)] = float(100. * cdr_generate_identity / n_sentences)
                    scores['%s-%s-%s_mt_cdr_gen_cdr_identity' % (data_set, lang1, lang2)] = float(100. * cdr_generate_cdr_identity / n_sentences)
                    scores['%s-%s-%s_mt_open_cdr_gen_identity' % (data_set, lang1, lang2)] = float(100. * open_cdr_generate_identity / n_sentences)
                    scores['%s-%s-%s_mt_open_cdr_gen_cdr_identity' % (data_set, lang1, lang2)] = float(100. * open_cdr_generate_cdr_identity / n_sentences)
                    


        else:
            scores['%s-%s_ae_acc' % (data_set, lang1)] = 100. * n_valid / n_words
            
        if lang1 != lang2:
            hyp_dir = os.path.join(params.hyp_path, str(scores['epoch']))
            # compute BLEU
            frw_name = 'frw{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            frw_path = os.path.join(hyp_dir, frw_name)

            # export sentences to hypothesis file / restore BPE segmentation
            with open(frw_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(forward_result) + '\n')
            restore_segmentation(frw_path)

            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(hyp_dir, hyp_name)

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # hypothesis / reference paths
            hyp_name = 'cdr_hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(hyp_dir, hyp_name)

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cdr_hypothesis) + '\n')
            restore_segmentation(hyp_path)

            hyp_name = 'open_cdr_hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(hyp_dir, hyp_name)

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cdr_open_hypothesis) + '\n')
            restore_segmentation(hyp_path)





    def calculate_identity(self, ref, gen_result, lang2, w2, beam_size=1):
        gen_identity, gen_cdr_identity = 0, 0
        
        
        for i in range(w2.shape[1]):
            beam_identity, beam_cdr_identity = 0, 0    
            for k in range(1, beam_size+1):
                gen_sent = gen_result[-(i*beam_size+k)].replace(" ", "")
                ref_sent = ref[-i].replace(" ", "")
                if len(gen_sent) == 0:
                    continue
                gen_alignment = self.aligner.align(ref_sent, gen_sent)[0]

                gen_matches = 0
                gen_index = 0
                gen_cdr_matches = 0

                for j in range(len(gen_alignment[0])):
                    if len(gen_sent) > 0 and j < len(gen_alignment[0]):
                        gen_matches += 1 if gen_alignment[0][j] == gen_alignment[1][j] else 0
                        if lang2 == 'ab':
                            gen_cdr_matches += 1 if w2[gen_index, -i] == 1 and gen_alignment[0][j] == gen_alignment[1][j] else 0
                            if gen_alignment[0][j] != '-':
                                gen_index += 1
                

                gen_matches /= len(ref_sent)

                gen_cdr_matches /= w2[:, -i].sum() - 1

                beam_identity += gen_matches
                beam_cdr_identity += gen_cdr_matches

            gen_identity += beam_identity/beam_size

            gen_cdr_identity += beam_cdr_identity/beam_size
            
        return gen_identity, gen_cdr_identity


def convert_to_text(batch, lengths, dico, params, mode='mt', map=None):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    if mode == 'mt':
        slen, bs = batch.shape
        assert lengths.max() == slen and lengths.shape[0] == bs
        assert (batch[0] == params.bos_index).sum() == bs or (batch[0] == params.eos_index).sum() == bs
        assert (batch == params.eos_index).sum() == bs
        sentences = []

        for j in range(bs):
            words = []
            for k in range(1, lengths[j]):
                if batch[k, j] == params.eos_index:
                    break
                if batch[k, j] < 5 or batch[k, j] > 24:
                    continue
                words.append(dico[batch[k, j]])
            sentences.append(" ".join(words))
    else:
        sentences = [[] for _ in range(len(lengths))]
        for i in range(len(map)):
            if batch[i] < 5 or batch[i] > 24:
                    continue
            sentences[map[i]].append(dico[batch[i]])

        sentences = [" ".join(words[1:-1]) for words in sentences]
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1
