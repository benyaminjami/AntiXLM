# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration


N_MAX_POSITIONS = 512  # maximum input sequence length

logger = getLogger()


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class TransformerModel(nn.Module):

    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs',
                  'n_words', 'dim', 'n_layers', 'n_heads', 'hidden_dim', 'dropout',
                  'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, is_encoder, with_output, encoder=None):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.params = params
        # dictionary / languages
        # self.n_langs = params.n_langs
        # self.n_words = params.n_words
        # self.eos_index = params.eos_index
        # self.pad_index = params.pad_index
        # self.id2lang = params.id2lang
        # self.lang2id = params.lang2id
        # self.lang_emb = params.lang_emb
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        self.model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-base")
        self.model.resize_token_embeddings(146)
        self.lang_tokens_ids = [144, 145]
        self.lang_tokens_dict = {'ab': 144, 'ag': 145}

    def forward(self, mode='fwd', **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        # TODO
        # elif mode == 'batch_fwd':
        #     x = kwargs['x']
        #     batch_size = 100
        #     n_batch = max(x.shape[1] // batch_size, 1)
        #     result = []
        #     for i in range(n_batch):
        #         result.append(self.fwd(x=kwargs['x1'][:,i*batch_size:(i+1)*batch_size],
        #                                len1=kwargs['len1'][i*batch_size:(i+1)*batch_size],
        #                                x2=kwargs['langs'][:,i*batch_size:(i+1)*batch_size],
        #                                causal=True,
        #                                src_enc=kwargs['src_enc'][i*batch_size:(i+1)*batch_size,:,:],
        #                                src_len=kwargs['src_len'][i*batch_size:(i+1)*batch_size],
        #                                # cache=kwargs['cache'],
        #                                # # bert_cache=kwargs['bert_cache'],
        #                                bert_embed=kwargs['bert_embed'][i*batch_size:(i+1)*batch_size,:,:]))
        #     return torch.cat(result, dim=1)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x1, len1, lang1, x2, len2, lang2, return_dict=False, **kwargs):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x1.size()
        assert len1.size(0) == bs
        assert len1.max().item() <= slen
        x1 = x1.clone().transpose(0, 1)  # batch size as dimension 0
        x2 = x2.clone().transpose(0, 1)  # batch size as dimension 0
        labels = x2.clone()

        src_lang = torch.ones((bs, 1)).type(torch.int) * self.lang_tokens_dict[lang1]
        tgt_lang = torch.ones((bs, 1)).type(torch.int) * self.lang_tokens_dict[lang2]

        x1 = torch.concatenate((src_lang.to(x1.device), tgt_lang.to(x1.device), x1), axis=1)
        labels = torch.concatenate((tgt_lang.to(x1.device), labels), axis=1)

        decoder_input_ids = self.model._shift_right(labels)

        labels[labels == 0] = -100
        labels[labels == 2] = -100

        slen += 2

        # generate masks
        mask, attn_mask = get_masks(slen=slen, lengths=len1+2, causal=False)

        output = self.model(input_ids=x1,
                            attention_mask=mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=None,
                            head_mask=None,
                            decoder_head_mask=None,
                            cross_attn_head_mask=None,
                            encoder_outputs=kwargs.get('encoder_outputs', None),
                            past_key_values=None,
                            inputs_embeds=None,
                            decoder_inputs_embeds=None,
                            labels=labels.contiguous(),
                            use_cache=True,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=True)

        if return_dict:
            return output

        return output['loss']

    def generate(self, x1, len1, lang1, lang2, num_beams=3, **kwargs):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <BOS> W1 W2 W3 <EOS> <PAD>
                <BOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        `sampling_method`:
            - G: Greedy
            - B: Beam sampling -> sampling_param: Beam size
            - T: Tempreture sampling -> sampling_param: tempreture
        """
        slen, bs = x1.size()
        assert len1.size(0) == bs
        assert len1.max().item() <= slen
        x1 = x1.clone().transpose(0, 1)  # batch size as dimension 0

        src_lang = torch.ones((bs, 1)).type(torch.int) * self.lang_tokens_dict[lang1]
        tgt_lang = torch.ones((bs, 1)).type(torch.int) * self.lang_tokens_dict[lang2]

        x1 = torch.concatenate((src_lang.to(x1.device), tgt_lang.to(x1.device), x1), axis=1)

        slen += 2
        # generate masks
        mask, attn_mask = get_masks(slen=slen, lengths=len1+2, causal=False)

        output = self.model.generate(input_ids=x1,
                                     attention_mask=mask,
                                     max_length=self.params.max_len[lang2],
                                     num_beams=num_beams,
                                     forced_bos_token_id=self.lang_tokens_dict[lang2],
                                     return_dict_in_generate=False, do_sample=True, **kwargs)

        lens = ((output != 0) & (output != 1)  & (output != self.lang_tokens_dict[lang2])).sum(1)

        # TODO
        output = output[:, 2:]
        
        if output.shape[1]-1 != lens.max():
            output = torch.concatenate((output, torch.zeros((bs, 1)).type(torch.int).to(x1.device)), axis=1)
            for i in range(bs):
                output[i, lens[i]] = 1
        lens = lens + 1
            
        return output, lens
    

    def open_end_weight(self, beam_w):
        for i in range(beam_w.shape[1]):
            flag = False
            j = 1
            while True:
                if not flag and beam_w[-j, i] == 0:
                    beam_w[-j, i] = 1
                elif not flag and beam_w[-j, i] == 1:
                    flag = True
                elif flag and beam_w[-j, i] == 1:
                    break
                j += 1
        return beam_w


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # if hyp[-1] == 0:
        #     return
        if len(self) < self.n_hyp or score > self.worst_score:

            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty
