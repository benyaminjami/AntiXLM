# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random
import argparse
import Ankh.utils as utils
import torch
from contextlib import nullcontext

from Ankh.slurm import init_signal_handler, init_distributed_mode
from Ankh.data.loader import check_data_params, load_data
from Ankh.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from Ankh.model import check_model_params, build_model
from Ankh.trainer import EncDecTrainer
from Ankh.evaluation.evaluator import EncDecEvaluator


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument(
        "--cuda", type=bool_flag, default=False, help="Is GPU availble?"
    )
    parser.add_argument(
        "--dump_path", type=str, default="./dumped/", help="Experiment dump path"
    )
    parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=10000,
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument("--exp_id", type=str, default="0", help="Experiment ID")
    parser.add_argument("--wandb_id", type=str, default="", help="wandb ID")

    # float16 / AMP API
    parser.add_argument(
        "--amp",
        type=int,
        default=1,
        help="""Use AMP wrapper for float16 / distributed / gradient accumulation.
                        Level of optimization. -1 to disable.""",
    )

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument(
        "--encoder_only", type=bool_flag, default=False, help="Only use an encoder"
    )

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512, help="Embedding layer size")

    parser.add_argument(
        "--n_layers", type=int, default=None, help="Number of Transformer layers"
    )
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=4,
        help="Number of Transformer layers for Encoder",
    )
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=4,
        help="Number of Transformer layers for Decoder",
    )

    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of Transformer heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--drop_net", type=float, default=1.0, help="Drop_net Prob")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--gelu_activation",
        type=bool_flag,
        default=True,
        help="Use a GELU activation instead of ReLU",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    parser.add_argument(
        "--sinusoidal_embeddings",
        type=bool_flag,
        default=False,
        help="Use sinusoidal embeddings",
    )
    parser.add_argument(
        "--lang_emb",
        type=str,
        default="emb",
        choices=["emb", "layer"],
        help="how to distinguish langs",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="B",
        choices=["B", "T", "G"],
        help="""
                            - G: Greedy
                            - B: Beam sampling -> sampling_param: Beam size, ex: 5/10/20
                            - T: Tempreture sampling -> sampling_param: tempreture ex: 0.5/1/2
                        """,
    )
    parser.add_argument("--sampling_param", type=int, default="5")
    parser.add_argument(
        "--fused_bert",
        type=bool_flag,
        default=False,
        help="INCORPORATING BERT INTO NEURAL MACHINE TRANSLATION",
    )

    # memory parameters
    parser.add_argument(
        "--use_memory", type=bool_flag, default=False, help="Use an external memory"
    )

    # adaptive softmax
    parser.add_argument(
        "--asm", type=bool_flag, default=False, help="Use adaptive softmax"
    )
    if parser.parse_known_args()[0].asm:
        parser.add_argument(
            "--asm_cutoffs",
            type=str,
            default="8000,20000",
            help="Adaptive softmax cutoffs",
        )
        parser.add_argument(
            "--asm_div_value",
            type=float,
            default=4,
            help="Adaptive softmax cluster sizes ratio",
        )

    # causal language modeling task parameters
    parser.add_argument(
        "--context_size",
        type=int,
        default=0,
        help="Context size (0 means that the first elements in sequences won't have any context)",
    )

    # masked language modeling task parameters
    parser.add_argument(
        "--word_pred",
        type=float,
        default=0.15,
        help="Fraction of words for which we need to make a prediction",
    )
    parser.add_argument(
        "--sample_alpha",
        type=float,
        default=0,
        help="Exponent for transforming word counts to probabilities (~word2vec sampling)",
    )
    parser.add_argument(
        "--word_mask_keep_rand",
        type=str,
        default="0.8,0.1,0.1",
        help="Fraction of words to mask out / keep / randomize, among the words to predict",
    )

    # input sentence noise
    parser.add_argument(
        "--word_shuffle",
        type=float,
        default=3,
        help="Randomly shuffle input words (0 to disable)",
    )
    parser.add_argument(
        "--word_dropout",
        type=float,
        default=0.1,
        help="Randomly dropout input words (0 to disable)",
    )
    parser.add_argument(
        "--word_blank",
        type=float,
        default=0.1,
        help="Randomly blank input words (0 to disable)",
    )

    # data
    parser.add_argument("--data_path", type=str, default="./data", help="Data path")
    parser.add_argument(
        "--lgs",
        type=str,
        default="at-ag",
        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)",
    )
    parser.add_argument(
        "--max_vocab",
        type=int,
        default=-1,
        help="Maximum vocabulary size (-1 to disable)",
    )
    parser.add_argument(
        "--min_count", type=int, default=0, help="Minimum vocabulary count"
    )
    parser.add_argument(
        "--lg_sampling_factor", type=float, default=-1, help="Language sampling factor"
    )
    parser.add_argument(
        "--skip_fw", type=bool_flag, default=False, help="Training on Frameworks or not"
    )
    parser.add_argument(
        "--cdr_weight",
        type=int,
        default=2,
        help="Weight that the cdr losses will be multipled with",
    )

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256, help="Sequence length")
    parser.add_argument(
        "--max_len",
        type=str,
        default="100,100",
        help="Maximum length of sentences for languages(after BPE)",
    )
    parser.add_argument(
        "--min_len",
        type=str,
        default="20,20",
        help="Minimum length of sentences for languages(after BPE)",
    )
    parser.add_argument(
        "--group_by_size",
        type=bool_flag,
        default=True,
        help="Sort sentences by size during the training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=0,
        help="""Maximum number of sentences per batch (used in combination with tokens_per_batch,
                        0 to disable)""",
    )
    parser.add_argument(
        "--tokens_per_batch", type=int, default=-1, help="Number of tokens per batch"
    )

    # training parameters
    parser.add_argument(
        "--split_data",
        type=bool_flag,
        default=True,
        help="Split data across workers of a same node",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam,lr=0.0001",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=100000,
        help="Epoch size / evaluation frequency (-1 for parallel data size)",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Maximum epoch size"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )
    parser.add_argument(
        "--validation_metrics", type=str, default="", help="Validation metrics"
    )
    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=2,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )

    # training coefficients
    parser.add_argument(
        "--lambda_mlm", type=str, default="1", help="Prediction coefficient (MLM)"
    )
    parser.add_argument(
        "--lambda_clm", type=str, default="1", help="Causal coefficient (LM)"
    )
    parser.add_argument("--lambda_pc", type=str, default="1", help="PC coefficient")
    parser.add_argument(
        "--lambda_ae",
        type=str,
        default="0:1,100000:0.1,300000:0",
        help="AE coefficient",
    )
    parser.add_argument("--lambda_mt", type=str, default="1", help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1", help="BT coefficient")

    # training steps
    parser.add_argument(
        "--clm_steps", type=str, default="", help="Causal prediction steps (CLM)"
    )
    parser.add_argument(
        "--mlm_steps", type=str, default="", help="Masked prediction steps (MLM / TLM)"
    )
    parser.add_argument(
        "--mt_steps", type=str, default="", help="Machine translation steps"
    )
    parser.add_argument(
        "--mt_steps_ratio", type=int, default=0, help="Machine translation steps ratio"
    )
    parser.add_argument(
        "--mt_steps_warmup",
        type=int,
        default=-1,
        help="Machine translation number of warmup steps",
    )
    parser.add_argument(
        "--ae_steps", type=str, default="ab,ag", help="Denoising auto-encoder steps"
    )
    parser.add_argument(
        "--bt_steps",
        type=str,
        default="at-ag-at,ag-at-ag",
        help="Back-translation steps",
    )
    parser.add_argument(
        "--pc_steps", type=str, default="", help="Parallel classification steps"
    )

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument(
        "--reload_emb", type=str, default="", help="Reload pretrained word embeddings"
    )
    parser.add_argument(
        "--reload_model", type=str, default="", help="Reload a pretrained model"
    )
    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )

    # beam search (for MT only)
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1,
        help="""Length penalty, values < 1.0 favor shorter sentences,
                        while values > 1.0 favor longer ones.""",
    )
    parser.add_argument(
        "--early_stopping",
        type=bool_flag,
        default=False,
        help="""Early stopping, stop as soon as we have `beam_size` hypotheses,
                        although longer ones may have better scores.""",
    )

    # evaluation
    parser.add_argument(
        "--eval",
        type=bool_flag,
        default=False,
        help="Evaluate BLEU score during MT training",
    )
    parser.add_argument(
        "--open",
        type=bool_flag,
        default=True,
        help="Generate CDR with unspecified length",
    )
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )

    # debug
    parser.add_argument(
        "--debug_train",
        type=bool_flag,
        default=False,
        help="Use valid sets for train sets (faster loading)",
    )
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    # multi-gpu / multi-node
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )

    return parser


def main(params):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # TODO
    # build model
    # model = build_model(params)

    # # load data
    # data = load_data(params)

    # TODO
    # build trainer, reload potential checkpoints / build evaluator
    trainer = EncDecTrainer(params)
    data = trainer.data
    # TODO
    evaluator = EncDecEvaluator(trainer, data, params)
    logger.info("Models Created")

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        for k in scores:
            utils.board_writer.add_scalar(
                str(k), float(scores[k]), trainer.n_total_iter
            )
            utils.wandb.log({str(k): float(scores[k])}, step=trainer.n_total_iter)

        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # language model training
    for _ in range(params.max_epoch):
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:
            # denoising auto-encoder steps
            if trainer.n_total_iter >= params.mt_steps_warmup:
                for lang in shuf_order(params.ae_steps):
                    with trainer.model.no_sync() if params.multi_gpu and False else nullcontext():
                        trainer.mt_step(lang, lang, params.lambda_ae)

            # machine translation steps
            if params.mt_steps_ratio != 0:
                if (
                    trainer.n_total_iter % params.mt_steps_ratio == 0
                    or trainer.n_total_iter < params.mt_steps_warmup
                ):
                    for lang1, lang2 in shuf_order(params.mt_steps):
                        with trainer.model.no_sync() if params.multi_gpu and False else nullcontext():
                            trainer.mt_step(lang1, lang2, params.lambda_mt)

            # back-translation steps
            if trainer.n_total_iter >= params.mt_steps_warmup:
                for i, (lang1, lang2, lang3) in enumerate(shuf_order(params.bt_steps)):
                    if i == 0:
                        with trainer.model.no_sync() if params.multi_gpu and False else nullcontext():
                            trainer.bt_step(
                                lang1, lang2, lang3, params.lambda_bt, i == 1
                            )
                            torch.cuda.empty_cache()
                    else:
                        trainer.bt_step(lang1, lang2, lang3, params.lambda_bt, i == 1)
                        torch.cuda.empty_cache()

            # logger.warning('END of Iteration')
            trainer.iter()
            trainer.save_periodic()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        if params.multi_gpu:
            torch.distributed.barrier()

        # evaluate perplexity
        scores = evaluator.run_all_evals(trainer)

        # # print / JSON log
        if params.reporter:
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))
            for k in scores:
                utils.board_writer.add_scalar(
                    str(k), float(scores[k]), trainer.n_total_iter
                )
                utils.wandb.log({str(k): float(scores[k])}, step=trainer.n_total_iter)

            if params.is_master:
                logger.info("__log__:%s" % json.dumps(scores))

            logger.info("*********")
            for k, v in trainer.n_pros_sents.items():
                logger.info(
                    "Number of processed sentences in {0} -> {1}".format(k, str(v))
                )
            logger.info("*********")

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        if params.multi_gpu:
            torch.distributed.barrier()


if __name__ == "__main__":
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # # debug mode
    if params.debug:
        params.exp_name = "debug"
        params.exp_id = "debug_%08i" % random.randint(0, 100000000)
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
