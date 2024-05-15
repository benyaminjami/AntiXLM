# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import Bio
import wandb
import subprocess
from Bio.Align import substitution_matrices
from collections import OrderedDict
import numpy as np
import torch
import xlm.utils as utils
from ..model.transformer import get_masks
from ..utils import to_cuda, restore_segmentation, concat_batches
from Bio import pairwise2
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
                group_by_size=False,
                n_sentences=n_sentences,
                beam_size=self.params.beam_size
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
        logger.info("============ Starting evaluating epoch %i ... ============" % trainer.epoch)
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})
        data_set_dictionary = {
            "ab-ag.test":0,
            "ag-ab.test":1,
            # "ag-ab.test-cdr":, 
            # "ag-ab.test-cdr_open":3,
            "ab-ag.valid":2,
            "ag-ab.valid":3,
            # "ag-ab.valid-cdr":6, 
            # "ag-ab.valid-cdr_open":7 
        }

        os.makedirs(os.path.join(params.hyp_path, str(scores['epoch'])), exist_ok=True)

        with torch.no_grad():

            for data_set in ['valid', 'test']:

                # for lang in params.ae_steps:
                #     self.evaluate_mt(scores, data_set, lang, lang, False)

                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in (('ag', 'ab'), ('ab', 'ag')):                    
                    generate = params.eval and \
                        data_set_dictionary["{0}-{1}.{2}".format(lang1, lang2, data_set)] % (params.world_size) == params.global_rank 
                    if generate:
                        self.evaluate_mt(scores, data_set, lang1, lang2, generate)

                # lang1, lang2 = 'ag', 'ab'              
                # generate = params.eval and \
                #     data_set_dictionary["{0}-{1}.{2}-cdr".format(lang1, lang2, data_set)] % (params.world_size) == params.global_rank 
                # if generate:
                #     self.evaluate_mt(scores, data_set, lang1, lang2, generate, cdr=True)
                
                
                # generate = params.eval and params.open and \
                #     data_set_dictionary["{0}-{1}.{2}-cdr_open".format(lang1, lang2, data_set)] % (params.world_size) == params.global_rank 
                # if generate:
                #     self.evaluate_mt(scores, data_set, lang1, lang2, generate, cdr=True, unrestricted=True)
                    
        logger.info("============ End evaluating epoch %i ... ============" % trainer.epoch)
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
        self.aligner.open_gap_score = -10
        self.aligner.extend_gap_score = -0.5


    def evaluate_mt(self, scores, data_set, lang1, lang2, generate, cdr=False):
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
        forward_identity, forward_cdr_identity = 0, 0

        # store hypothesis to compute BLEU score


        hypothesis = []
        forward_result = []
  
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
            n_sentences += x1.shape[1]
            n_words += y.size(0)
            token_type_ids = torch.zeros_like(x1)
            
            # cuda
            # TODO: GPU
            if self.params.cuda:
                x1, len1, langs1, x2, len2, langs2, y, token_type_ids = to_cuda(x1, len1, langs1, x2, len2, langs2, y, token_type_ids)
            ref = convert_to_text(x2, len2, self.dico, params)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.params.amp == 1)):
                bert_embed = None
                if self.fused_bert:    
                    bert_embed = self.bert(input_ids=x1.T, token_type_ids=token_type_ids.T, attention_mask=get_masks(x1.size()[0], len1, False)[0]).last_hidden_state
                # encode source sentence
                enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False, bert_embed=bert_embed)
                enc1 = enc1.transpose(0, 1)

                # decode target sentence
                #TODO
                if generate:
                    dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1, bert_embed=bert_embed)

                    # loss
                    word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

                    # update stats
                    xe_loss += loss.item() * len(y)
                    n_valid += (word_scores.max(1)[1] == y).sum().item()

                    forward_text = convert_to_text(word_scores.max(1)[1],
                        len2, 
                        self.dico, 
                        params, 
                        'ae',
                        torch.arange(len(len2), dtype=torch.long, device=pred_mask.device).repeat(pred_mask.shape[0],1).masked_select(pred_mask))
                    
                    forward_result.extend(forward_text)

                    batch_forward_identity, batch_forward_cdr_identity = sequence_identity(ref, forward_text, w2.transpose(0,1).tolist())
                    forward_identity += batch_forward_identity
                    forward_cdr_identity += batch_forward_cdr_identity
                    
                    report_metric(metric_name='mt_ppl', step=scores['epoch'], value=np.exp(xe_loss / n_words), args=(data_set, lang1, lang2), scores=scores)

                    report_metric(metric_name='mt_acc', step=scores['epoch'], value=float(100. * n_valid / n_words), args=(data_set, lang1, lang2), scores=scores)

                    report_metric(metric_name='mt_frw_identity', step=scores['epoch'], value=float(100. * forward_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)

                    report_metric(metric_name='mt_frw_cdr_identity', step=scores['epoch'], value=float(100. * forward_cdr_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)

                 
                if generate:
                    if params.beam_size == 1:
                        generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=self.params.max_len[lang2], bert_embed=bert_embed)
                        
                    else:
                        generated, lengths = decoder.generate_beam(
                            enc1, len1, lang2_id, beam_size=params.beam_size,
                            length_penalty=params.length_penalty,
                            early_stopping=params.early_stopping,
                            max_len=self.params.max_len[lang2] + 2,
                            bert_embed=bert_embed,
                            cdr_generation=cdr,
                            tgt_frw=x2,
                            w=w2, 
                        )
                        
                    hypothesis_text = convert_to_text(generated, lengths, self.dico, params)
                    hypothesis.extend(hypothesis_text)
                    batch_generate_identity, batch_generate_cdr_identity = sequence_identity(ref, hypothesis_text, w2.transpose(0,1).tolist())
                    generate_identity += batch_generate_identity
                    generate_cdr_identity += batch_generate_cdr_identity

                    report_metric(metric_name='mt_gen_identity', step=scores['epoch'], value=float(100. * generate_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)

                    report_metric(metric_name='mt_gen_cdr_identity', step=scores['epoch'], value=float(100. * generate_cdr_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)



        s = ''
        if cdr:
            s+='fixedFW_'
            
        # if lang1 != lang2:


        #     if generate:    
        #         report_metric(metric_name='mt_frw_identity', step=scores['epoch'], value=float(100. * forward_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)
 
        #         report_metric(metric_name='mt_gen_identity', step=scores['epoch'], value=float(100. * generate_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)
                
        #         if lang2 == 'ab':
        #             report_metric(metric_name='mt_frw_cdr_identity', step=scores['epoch'], value=float(100. * forward_cdr_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)
                    
        #             report_metric(metric_name='mt_gen_cdr_identity', step=scores['epoch'], value=float(100. * generate_cdr_identity / n_sentences), args=(data_set, lang1, lang2), scores=scores)
                    

        # else:
        #     report_metric(metric_name='ae_acc', step=scores['epoch'], value=float(100. * n_valid / n_words), args=(data_set, lang1), scores=scores)
            
        if lang1 != lang2:
            hyp_dir = os.path.join(params.hyp_path, str(scores['epoch']))
            frw_name = 'frw{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            frw_path = os.path.join(hyp_dir, frw_name)

            with open(frw_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(forward_result) + '\n')
            restore_segmentation(frw_path)

            hyp_name = 'hyp{0}.{1}-{2}.{3}.{4}.txt'.format(scores['epoch'], lang1, lang2, data_set, s)
            hyp_path = os.path.join(hyp_dir, hyp_name)

            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)


from Bio import pairwise2
import numpy as np


def sequence_identity(references, strings, masks):
    if len(references) != len(strings):
        raise ValueError("The number of reference sequences and strings must be equal.")

    def cal_identity(alignment, mask):
        seq1, seq2 = alignment[0], alignment[1]
        matches = np.array([a == b for a, b in zip(seq1, seq2)])
        if mask is not None:
            n_matches = 0
            j = 0
            for i, a in enumerate(seq1):
                if seq1[i] == seq2[i] and mask[j] == 1:
                    n_matches += 1
                if a != '-':
                    j+=1    
            return (n_matches / sum(mask))
            
        else:
            return matches.sum()/len(seq1)


    blosum62 = substitution_matrices.load("BLOSUM62")
    total_identity = 0
    total_region_identity = 0
        
    for reference, string, mask in zip(references, strings, masks):
        if string == '':
            continue
        alignment = pairwise2.align.globalds(reference.replace(' ',''), string.replace(' ',''), blosum62, -10, -0.5)[0]
        identity = cal_identity(alignment, None)
        total_identity += identity

        region_identity = cal_identity(alignment, mask)
        total_region_identity += region_identity
        
    average_identity = total_identity / len(strings)
    average_region_identity = total_region_identity / len(strings)
    return average_identity, average_region_identity


def calculate_identity(aligner, ref, gen_result, lang2, w2, beam_size=1):
    gen_identity, gen_cdr_identity = 0, 0
    
    
    for i in range(w2.shape[1]):
        beam_identity, beam_cdr_identity = 0, 0    
        for k in range(1, beam_size+1):
            gen_sent = gen_result[-(i*beam_size+k)].replace(" ", "")
            ref_sent = ref[-i].replace(" ", "")
            if len(gen_sent) == 0:
                continue
            # TODO
            gen_alignment = pairwise2.align.globaldx(ref_sent, gen_sent, substitution_matrices.load("BLOSUM62"))[0]
            # gen_alignment = [ref_sent, gen_sent]

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

            gen_cdr_matches /= w2[:, -i].sum()

            beam_identity += gen_matches
            beam_cdr_identity += gen_cdr_matches

        gen_identity += beam_identity/beam_size

        gen_cdr_identity += beam_cdr_identity/beam_size
        
    return gen_identity, gen_cdr_identity

def report_metric(metric_name, step, value, args, scores):
    metric_str = '-'.join(['%s']*len(args)) + '_' + metric_name
    metric_str = metric_str % args
    scores[metric_str] = value
    utils.wandb.log({metric_str: value}, step=step)
    logger.info("%s -> %.6f" % (metric_str, value))

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
            for k in range(0, lengths[j]):
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

        sentences = [" ".join(words) for words in sentences]
    return sentences

# def calculate_identity(aligner, ref, gen_result, lang2, w2, beam_size=1):
#     gen_identity, gen_cdr_identity = 0, 0
#     gen_cdr123_identity = {1:0,2:0,3:0}
    
#     for i in range(w2.shape[1]):
#         beam_identity, beam_cdr_identity = 0, 0    
#         for k in range(1, beam_size+1):
#             gen_sent = gen_result[-(i*beam_size+k)].replace(" ", "")
#             ref_sent = ref[-i].replace(" ", "")
#             if len(gen_sent) == 0:
#                 continue
#             gen_alignment = aligner.align(ref_sent, gen_sent)[0]

#             gen_matches = 0
#             gen_index = 0
#             gen_cdr_matches = 0
#             cdr_number = 0
#             cdr_flag = False
#             gen_cdr123_matches = {1:0, 2:0, 3:0}
#             gen_cdr123_len = {1:0, 2:0, 3:0}

#             for j in range(len(gen_alignment[0])):
#                 if len(gen_sent) > 0 and j < len(gen_alignment[0]):
#                     gen_matches += 1 if gen_alignment[0][j] == gen_alignment[1][j] else 0
#                     if lang2 == 'ab':
#                         if w2[gen_index, -i] == 1 and not cdr_flag:
#                             cdr_number += 1
#                             cdr_flag = True
#                         elif w2[gen_index, -i] == 0 and cdr_flag:
#                             cdr_flag = False
#                         if cdr_flag:
#                             gen_cdr123_len[cdr_number] += 1
#                             gen_cdr123_matches[cdr_number] += 1 if gen_alignment[0][j] == gen_alignment[1][j] else 0 

#                         gen_cdr_matches += 1 if w2[gen_index, -i] == 1 and gen_alignment[0][j] == gen_alignment[1][j] else 0
#                         if gen_alignment[0][j] != '-':
#                             gen_index += 1
            

#             gen_matches /= len(ref_sent)

#             gen_cdr_matches /= w2[:, -i].sum() - 1

#             beam_identity += gen_matches
#             beam_cdr_identity += gen_cdr_matches

#         gen_identity += beam_identity/beam_size

#         gen_cdr_identity += beam_cdr_identity/beam_size
        
#     return gen_identity, gen_cdr_identity
