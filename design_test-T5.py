import os
import argparse
import copy
import json
import pickle
import Bio
from Bio import Align

from Bio.Align import substitution_matrices
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from abnumber import Chain
from evaluation.datasets import SAbDabDataset
from evaluation.datasets import get_dataset
from evaluation.utils.protein.writers import save_pdb
from evaluation.utils.data import *
from evaluation.utils.misc import *
from evaluation.utils.transforms import *
from Ankh.utils import AttrDict
from Ankh.utils import bool_flag, initialize_exp
from Ankh.data.dictionary import Dictionary
from Ankh.model.transformer import TransformerModel
from Ankh.utils import to_cuda
from Ankh.model.transformer import get_masks
from Ankh.evaluation.evaluator import convert_to_text
from transformers import BertModel, T5ForConditionalGeneration, AutoTokenizer


import pyrosetta
pyrosetta.init(silent=True)

from pyrosetta import pose_from_pdb, init
# from pyrosetta.rosetta import *
# from pyrosetta.teaching import *

#Core Includes
from rosetta.core.select import residue_selector as selections

from rosetta.protocols import antibody
init('-use_input_sc -ignore_unrecognized_res -check_cdr_chainbreaks false \
     -ignore_zero_occupancy false -load_PDB_components false -no_fconfig', silent=True)


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="kir", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="123", help="Experiment ID")
    parser.add_argument("--beam_size", type=int, default=100)
    parser.add_argument("--eval_modes", type=list, default=['CDR3'])
    parser.add_argument("--excess_res", type=int, default=50)
    parser.add_argument("--reporter", type=bool, default=False)
    # model / output paths
    parser.add_argument("--model_path", type=str, default="/checkpoint/benjami/10635261/unsupMT_agab/0/checkpoint.pth", help="Model path")
    parser.add_argument("--output_path", type=str, default="evaluation/", help="Output path")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="ag", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="ab", help="Target language")

    parser.add_argument('-i', '--index', type=int, default=0)
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-c', '--config', type=str, default='evaluation/configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='evaluation/results')

    return parser

class SabdabEntry:
    def __init__(self, dataset, index, params, renumber=None) -> None:
        self.structure = dataset[index]
        self.entry = self.find_entry(dataset, index)
        
        self.structure_id = self.structure['id']
        self.ag_name = self.entry['ag_name']
        self.ag_chain = self.entry['ag_chains'][0]
        self.ab_chain = self.entry['H_chain']
        self.pdb_code = self.entry['pdbcode']

        self.f1 = self.structure['heavy']['FW1_seq']
        self.f2 = self.structure['heavy']['FW2_seq']
        self.f3 = self.structure['heavy']['FW3_seq']
        self.f4 = self.structure['heavy']['FW4_seq']

        self.c1 = self.structure['heavy']['H1_seq']
        self.c2 = self.structure['heavy']['H2_seq']
        self.c3 = self.structure['heavy']['H3_seq']
        self.excess = params.excess_res

        self.ab_seq = self.structure['heavy'].seq
        self.ag_seq = self.structure['antigen'].seq
        self.weights = {}
        self.set_weights()
        data_native = MergeChains()(self.structure)
        self.log_dir = get_new_log_dir(os.path.join(params.log_dir), prefix='%02d_%s' % (index, self.structure_id))
        save_pdb(data_native, os.path.join(self.log_dir, 'reference.pdb'))
        save_pdb(data_native, os.path.join(self.log_dir, 'reference_renamed.pdb'), 
                 rename={self.ag_chain:'A', self.ab_chain:'H'})
        pose = pose_from_pdb(os.path.join(self.log_dir, 'reference_renamed.pdb'))
        ab_info = antibody.AntibodyInfo(pose, antibody.Chothia_Scheme, antibody.North)

        for s in range(5,25):
            self.epi_residues = np.array(antibody.select_epitope_residues(ab_info, pose, s))[len(self.ab_seq):]
            if self.epi_residues.sum() > 20:
                break
        self.epi_range = (np.argmax(self.epi_residues), self.epi_residues.shape[0] - np.argmax(self.epi_residues[::-1]) - 1)
        self.epi_resseq = self.structure['antigen']['resseq'][self.epi_residues]

        save_pdb(data_native, os.path.join(self.log_dir, 'antigen.pdb'), ignore_chain=self.ab_chain)
        save_pdb(data_native, os.path.join(self.log_dir, 'antibody.pdb'), ignore_chain=self.ag_chain)        
        save_pdb(data_native, os.path.join(self.log_dir, 'cutted_antigen.pdb'), ignore_chain=self.ab_chain,
                 write_range={self.ag_chain: (max(0, self.epi_range[0]-self.excess), self.epi_range[1]+self.excess)})        
        save_pdb(data_native, os.path.join(self.log_dir, 'cutted_refrence_renamed.pdb'),
                 write_range={'A': (max(0, self.epi_range[0]-self.excess), self.epi_range[1]+self.excess)},
                 rename={self.ag_chain:'A', self.ab_chain:'H'})

        self.identity = {}
        self.generated_sequences = {}

    @property
    def antigen(self):
        extra = int(min(self.excess, (200 - (self.epi_range[1] - self.epi_range[0]))/2))
        return self.ag_seq[max(0, self.epi_range[0]-extra): self.epi_range[1]+extra]

    @property
    def antibody(self):
        return self.ab_seq

    def set_weights(self):
        self.weights['CDR1'] = self._construct_weight(cdr1=True)
        self.weights['CDR2'] = self._construct_weight(cdr2=True)
        self.weights['CDR3'] = self._construct_weight(cdr3=True)
        self.weights['CDR123'] = self._construct_weight(cdr1=True, cdr2=True, cdr3=True)

    def _construct_weight(self, cdr1=False, cdr2=False, cdr3=False):
        return [0] * len(self.f1) + \
            ([1] * len(self.c1) if cdr1 else [0] * len(self.c1)) + \
            [0] * len(self.f2) + \
            ([1] * len(self.c2) if cdr2 else [0] * len(self.c2)) + \
            [0] * len(self.f3) + \
            ([1] * len(self.c3) if cdr3 else [0] * len(self.c3)) + \
            [0] * len(self.f4)


    def find_entry(self, dataset:SAbDabDataset, index):
        for entry in dataset.sabdab_entries:
            if entry['id'] == self.structure['id']:
                return entry

    def write_generated(self):
        for key in self.generated_sequences:
        # Create a file name for the fasta file
            with open(os.path.join(self.log_dir, key)+'.fasta', "w") as f:
                for i, seq in enumerate(self.generated_sequences[key]):
                    # Open the file for writing
                    f.write(">{0}_sequence".format(key) + str(i) + "\n")
                    # Write the sequence to the file in fasta format
                    f.write(seq.replace(' ', '') + "\n")

    

def get_model(params):    
    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    
    model = T5ForConditionalGeneration.from_pretrained("ElnaggarLab/ankh-base")
    lang_tokens_dict = {'ab': 144, 'ag': 145}
    reloaded["model"] = {
                k.replace("module.model.", ""): reloaded["model"][k] for k in reloaded["model"]
            }
    model.resize_token_embeddings(146)

    tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-base")

        # reload model parameters
    model.load_state_dict(reloaded["model"])
    
    params.src_id = model_params.lang2id['ag']
    params.tgt_id = model_params.lang2id['ab']
    params.src_lang = 'ag'
    params.tgt_lang = 'ab'
    model.eval()
    
    
    return model, dico, lang_tokens_dict, tokenizer


def get_sabdab(params):
    # Load configs
    config, config_name = load_config(params.config)
    # Testset
    dataset = get_dataset(config.dataset.test)
    # Logging

    return dataset


def build_batch(seq, lang, eos, bos, pad):
    x1 = x1.clone().transpose(0, 1)  # batch size as dimension 0

    src_lang = torch.ones((1, 1)).type(torch.int) * lang_tokens_dict[lang1]
    tgt_lang = torch.ones((1, 1)).type(torch.int) * lang_tokens_dict[lang2]

    x1 = torch.concatenate((src_lang.to(x1.device), tgt_lang.to(x1.device), x1), axis=1)

    slen += 2
    # generate masks
    mask, attn_mask = get_masks(slen=slen, lengths=len1+2, causal=False)
    
    
    lengths = torch.LongTensor([len(seq) + 2])
    batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(pad)
    batch[0] = bos
    batch[1:lengths[0] - 1, 0].copy_(seq)
    batch[lengths[0] - 1, 0] = eos
    langs = batch.clone().fill_(lang)
    return batch, lengths, langs


def write_generated(log_dir, eval_step, sequences):
    log = get_new_log_dir(os.path.join(log_dir, eval_step))
    for i, seq in enumerate(sequences):
    # Create a file name for the fasta file
        filename = "sequence" + str(i) + ".fasta"

        # Open the file for writing
        with open(os.path.join(log, filename), "w") as f:
            # Write the sequence to the file in fasta format
            f.write(">sequence" + str(i) + "\n")
            f.write(seq.replace(' ', '') + "\n")

def apply_mask(string, mask):
    return ''.join([c for c, m in zip(string, mask) if m == 1])

def cal_identity(reference, alignment, mask):
    seq1, seq2 = alignment[0], alignment[1]
    if seq1.replace('-','') != reference:
        seq1, seq2 = seq2, seq1
    matches = np.array([a == b for a, b in zip(seq1, seq2)])
    if mask is not None:
        n_matches = 0
        j = 0
        for i, a in enumerate(seq1):
            if seq1[i] == seq2[i] and mask[j] == 1:
                n_matches += 1
            if a != '-':
                j+=1    
        return (n_matches / mask.sum()).item()
        
    else:
        return matches.sum()/len(seq1)

from Bio import pairwise2

def average_sequence_identity(reference, strings, mask):
    blosum62 = substitution_matrices.load("BLOSUM62")
    
    total_identity = 0
    total_region_identity = 0
    reference_region = apply_mask(reference, mask)
    
    for string in strings:
        alignment = pairwise2.align.globalds(reference, string, blosum62, -10, -0.5)[0]
        identity = cal_identity(reference, alignment, None)
        total_identity += identity

        # alignmentexo_region = pairwise2.align.globalds(reference_region, apply_mask(string, mask), blosum62, -10, -0.5)[0]
        region_identity = cal_identity(reference, alignment, mask)
        total_region_identity += region_identity
        
    average_identity = total_identity / len(strings)
    average_region_identity = total_region_identity / len(strings)
    return average_identity, average_region_identity


def evaluate(model, tokenizer, dico, lang_tokens_dict, params, aligner, sample:SabdabEntry, eval_modes=['CDR1', 'CDR2', 'CDR3', 'CDR123', 'GEN']):
    
    
    ag_tensor = torch.LongTensor([[dico.index(w) for w in sample.antigen]+[1]]).clone()  # batch size as dimension 0
    ag_length_tensor = torch.LongTensor([ag_tensor.shape[1]])
    
    bs, slen = ag_tensor.size()
    assert ag_length_tensor.size(0) == bs
    assert ag_length_tensor.max().item() <= slen
    
    src_lang = torch.ones((bs, 1)).type(torch.int) * lang_tokens_dict[params.src_lang]
    tgt_lang = torch.ones((bs, 1)).type(torch.int) * lang_tokens_dict[params.tgt_lang]

    ag_tensor = torch.concatenate((src_lang.to(ag_tensor.device), tgt_lang.to(ag_tensor.device), ag_tensor), axis=1)

    slen += 2
    # generate masks
    mask, attn_mask = get_masks(slen=slen, lengths=ag_length_tensor+2, causal=False)
    
    
    
    
    ag_tensor, ag_length_tensor = to_cuda(ag_tensor, ag_length_tensor)

    for eval_step in eval_modes:
        beam_size = params.beam_size

        ab_tensor = [[dico.index(w) for w in sample.antibody]+[1]]
        w = torch.LongTensor(sample.weights.get(eval_step, sample.weights['CDR123']))
        decoder_input = torch.LongTensor([[0, lang_tokens_dict[params.tgt_lang]] + ab_tensor[0][:w.argmax()]]).cuda()
        max_len = w.argmax() + sum(w)
        ab_tensor = torch.LongTensor(ab_tensor)
        with torch.no_grad(): 
            generated = model.generate(input_ids=ag_tensor,
                                     attention_mask=mask.to(ag_tensor.device),
                                     max_length=150,
                                     num_return_sequences=beam_size,
                                     num_beams=beam_size,
                                     forced_bos_token_id=lang_tokens_dict['ab'],
                                     decoder_input_ids=decoder_input.to(ag_tensor.device),
                                     return_dict_in_generate=False, do_sample=False)
                
                    
            hypothesis_text = convert_to_text(generated, tokenizer)
            batch_generate_identity, batch_generate_cdr_identity = average_sequence_identity(sample.antibody, hypothesis_text, w)
            print(batch_generate_identity, batch_generate_cdr_identity)
            # batch_generate_identity, batch_generate_cdr_identity = calculate_identity(aligner, [sample.antibody], hypothesis_text, 'ab', ab_weights_tensor, beam_size=beam_size)
            write_generated(sample.log_dir, eval_step, hypothesis_text)
            sample.generated_sequences[eval_step] = hypothesis_text
            
            if eval_step == 'GEN':
                sample.identity[eval_step+'_CDR'] = batch_generate_cdr_identity
                sample.identity[eval_step+'_ALL'] = batch_generate_identity
            else:
                sample.identity[eval_step] = batch_generate_cdr_identity

import sys
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def main():
    parser = get_parser()
    params = parser.parse_args()
    log = get_new_log_dir(os.path.join(params.out_root), date=True)
    params.log_dir = log
    dataset = get_sabdab(params)
    test_samples = []
    
    model, dico, lang_tokens_dict, tokenizer = get_model(params)
    model.cuda()
    lens = 0
    samples = dict()
    identity = {e:0 for e in params.eval_modes}
    for i in range(len(dataset.ids_in_split)):
        try:
            blockPrint()
            sample = SabdabEntry(dataset=dataset, index=i, params=params)
        except:
            continue
        enablePrint()
        test_samples.append(sample)
        evaluate(model, tokenizer, dico, lang_tokens_dict, params, None, sample, eval_modes=params.eval_modes)
        sample.write_generated()
        with open(os.path.join(sample.log_dir, 'sample.pkl'), 'wb') as f:
            pickle.dump(sample, f)
        print('###########################')
        print(i)
        print(sample.identity)
        print('###########################')
        structure_id = sample.structure_id.split('_')[0]
        samples[structure_id] = samples.get(structure_id, [])
        samples[structure_id].append(sample)
        
    # lens /= len(dataset.ids_in_split)
    # print(lens)
    total_identity = 0
    cnt = 0
    identity_samples = dict()
    for k in samples:
        t = {e:0 for e in params.eval_modes}
        for s in samples[k]:
            for e in t:
                t[e] += s.identity[e]
                total_identity += s.identity[e]
                cnt += 1
        identity_samples[k] = t
    
    print(identity_samples)

    for e in identity:
        for k in identity_samples:
            identity[e] += identity_samples[k][e] / len(samples[k])
        identity[e] /= len(samples)
        

    with open(os.path.join(params.log_dir, 'samples.pkl'), 'wb') as f:
        pickle.dump(test_samples, f)

    print('total')
    print(total_identity/cnt)
    print(identity)

if __name__ == '__main__':
    main()







