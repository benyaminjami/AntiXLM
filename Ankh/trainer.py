# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import Ankh.utils as utils
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

# import apex
from Ankh.data.loader import check_data_params, load_data
from Ankh.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from Ankh.model import check_model_params, build_model
from .optim import get_optimizer
from .utils import to_cuda
from .utils import parse_lambda_config, update_lambdas

# from .model.memory import HashingMemory

logger = getLogger()


class Trainer(object):
    def __init__(self, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        self.scaler = torch.cuda.amp.GradScaler()

        # data iterators
        self.iterators = {}
        self.model = build_model(params)
        # set parameters
        self.set_parameters()

        # self.set_optimizers()

        # self.reload_checkpoint()

        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")

            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[params.local_rank],
                output_device=params.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
            logger.info("Done nn.parallel.DistributedDataParallel")

        # TODO
        # logger.info("Compiling the Model ...")
        # self.encoder = torch.compile(self.encoder)
        # self.decoder = torch.compile(self.decoder)

        # set optimizers
        self.set_optimizers()

        # stopping criterion used for early stopping
        # if params.stopping_criterion != '':
        #     split = params.stopping_criterion.split(',')
        #     assert len(split) == 2 and split[1].isdigit()
        #     self.decrease_counts_max = int(split[1])
        #     self.decrease_counts = 0
        #     if split[0][0] == '_':
        #         self.stopping_criterion = (split[0][1:], False)
        #     else:
        #         self.stopping_criterion = (split[0], True)
        #     self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        # else:
        #     self.stopping_criterion = None
        #     self.best_stopping_criterion = None

        self.data = load_data(params)

        # probability of masking out / randomize / not modify words to predict
        params.pred_probs = torch.FloatTensor(
            [params.word_mask, params.word_keep, params.word_rand]
        )

        # probabilty to predict a word
        counts = np.array(list(self.data["dico"].counts.values()))
        params.mask_scores = np.maximum(counts, 1) ** -params.sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <PAD> index
        params.mask_scores[counts == 0] = 0  # do not predict special tokens

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m[1:], False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics
        }

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_pros_sents = {"AE_ab": 0, "AE_ag": 0, "BT_ab": 0, "BT_ag": 0}
        self.n_sentences = 0
        self.stats = OrderedDict(
            [("processed_s", 0), ("processed_w", 0)]
            + [("AE-%s-loss" % lang, []) for lang in params.ae_steps]
            + [("MT-%s-%s-loss" % (l1, l2), []) for l1, l2 in params.mt_steps]
            + [
                ("BT-%s-%s-%s-loss" % (l1, l2, l3), [])
                for l1, l2, l3 in params.bt_steps
            ]
        )

        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        self.parameters_name = []
        named_params = []
        named_params.extend(
            [(k, p) for k, p in self.model.named_parameters() if p.requires_grad]
        )

        # model (excluding memory values)
        self.parameters["model"] = [p for k, p in named_params]
        self.parameters_name = [k for k, p in named_params]

        # log
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizing_steps = []

        # model optimizer (excluding memory values)
        self.optimizer = get_optimizer(self.parameters["model"], params.optimizer)
        self.optimizer.zero_grad()

        # log
        # logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss, step, optimize=False):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")

            exit()

        params = self.params

        # optimizers
        acc_steps = len(
            self.params.ae_steps + self.params.bt_steps + self.params.mt_steps
        )
        # regular optimization
        if params.amp == 0:
            # without accumulation
            self.optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.parameters[name]])) ** 0.5
                # print(name, norm_check_a, norm_check_b)

            self.optimizer.step()
            return

            if (
                self.n_iter != 0
                and self.n_iter % params.accumulate_gradients == 0
                and optimize
            ):
                loss /= acc_steps
                loss.backward()
            else:
                # TODO
                # if params.clip_grad_norm > 0:
                #     clip_grad_norm_(self.parameter, params.clip_grad_norm)
                loss /= acc_steps
                loss.backward()

                flag1, flag2 = False, False
                zero_grads = ""
                for i in range(len(self.parameters["model"])):
                    if self.parameters["model"][i].grad is None:
                        logger.warning(
                            "No Gradient for layer {0}:   {1}".format(
                                i, self.parameters_name[i]
                            )
                        )
                        flag1 = True
                    elif self.parameters["model"][i].grad.count_nonzero() == 0:
                        zero_grads += str(i) + ": " + self.parameters_name[i] + "\t"
                        flag2 = True
                if flag1:
                    exit()
                if flag2:
                    logger.warning(str(self.optimizing_steps))
                    logger.warning("Zero Gradient for layers {0}".format(zero_grads))

                self.optimizer.step()
                self.optimizer.zero_grad()

        # AMP optimization
        else:
            if (
                self.n_iter != 0
                and self.n_iter % params.accumulate_gradients == 0
                and optimize
            ):
                self.scaler.scale(loss).backward()
                self.optimizing_steps.append(step)

                flag1, flag2 = False, False
                zero_grads = ""
                for i in range(len(self.parameters["model"])):
                    if self.parameters["model"][i].grad is None:
                        logger.warning(
                            "No Gradient for layer {0}:   {1}".format(
                                i, self.parameters_name[i]
                            )
                        )
                        flag1 = True
                    elif self.parameters["model"][i].grad.count_nonzero() == 0:
                        zero_grads += str(i) + ": " + self.parameters_name[i] + "\t"
                        flag2 = True
                if flag1:
                    exit()
                if flag2:
                    logger.warning(str(self.optimizing_steps))
                    logger.warning("Zero Gradient for layers {0}".format(zero_grads))

                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled,
                    # clips as usual:
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad()
                self.scaler.update()
                self.optimizing_steps.clear()

            else:
                if params.multi_gpu:
                    with self.model.no_sync():
                        self.scaler.scale(loss).backward()
                else:
                    self.scaler.scale(loss).backward()

                self.optimizing_steps.append(step)

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if not self.params.reporter:
            return
        if self.n_total_iter % 5 != 0:
            return

        for k, v in self.stats.items():
            if type(v) is list:
                if len(v) > 0:
                    utils.board_writer.add_scalar(
                        str(k), float(v[-1]), self.n_total_iter
                    )
                    utils.wandb.log({str(k): np.mean(v)}, step=self.n_total_iter)
            else:
                utils.board_writer.add_scalar(str(k), float(v), self.n_total_iter)
                utils.wandb.log({str(k): float(v)}, step=self.n_total_iter)

        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k, np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - "
        s_lr = (
            s_lr
            + (" - %s LR: " % k)
            + " / ".join(
                "{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups
            )
        )
        utils.wandb.log(
            {"Learning Rate": float(self.optimizer.param_groups[0]["lr"])},
            step=self.n_total_iter,
        )

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats["processed_s"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        utils.wandb.log(
            {
                "processed_s": float(self.stats["processed_s"] * 1.0 / diff),
                "processed_w": float(self.stats["processed_w"] * 1.0 / diff),
            },
            step=self.n_total_iter,
        )

        self.stats["processed_s"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time

        s_ns = " - N_Sentences: {}".format(self.n_sentences)

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr + s_ns)

    def get_iterator(self, iter_name, lang1, lang2, stream):
        """
        Create a new iterator for a dataset.
        """
        logger.info(
            "Creating new training data iterator (%s) ..."
            % ",".join([str(x) for x in [iter_name, lang1, lang2] if x is not None])
        )
        assert (
            stream or not self.params.use_memory or not self.params.mem_query_batchnorm
        )
        if lang2 is None:
            if stream:
                iterator = self.data["mono_stream"][lang1]["train"].get_iterator(
                    shuffle=True
                )
            else:
                iterator = self.data["mono"][lang1]["train"].get_iterator(
                    shuffle=True,
                    group_by_size=self.params.group_by_size,
                    n_sentences=-1,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data["para"][(_lang1, _lang2)]["train"].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(
            0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1))
        )
        noise[0] = -1  # do not move start sentence symbol

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[: l[i] - 1, i]
            permutation = scores.argsort()
            # shuffle words
            x2[: l[i] - 1, i].copy_(x2[: l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[: l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[: l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x == eos).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == eos
            words = x[: l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [
                w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)
            ]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[: l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "n_pros_sents": self.n_pros_sents
            # 'best_stopping_criterion': self.best_stopping_criterion,
        }

        logger.warning("Saving parameters ...")
        data["model"] = self.model.state_dict()

        if include_optimizers:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()

        data["dico_id2word"] = self.data["dico"].id2word
        data["dico_word2id"] = self.data["dico"].word2id
        data["dico_counts"] = self.data["dico"].counts
        data["params"] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        if not self.params.multi_gpu:
            data["model"] = {
                k.replace("module.", ""): data["model"][k] for k in data["model"]
            }

        # reload model parameters
        self.model.load_state_dict(data["model"])

        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])
        # # reload optimizers
        # for name in self.optimizers.keys():
        #      if False:
        # # AMP checkpoint reloading is buggy, we cannot do that - TODO:fix - https://github.com/NVIDIA/apex/issues/250
        #          logger.warning(f"Reloading checkpoint optimizer {name} ...")
        #          self.optimizers[name].load_state_dict(data[f'{name}_optimizer'])
        #      else:  # instead, we only reload current iterations / learning rates
        #          logger.warning(f"Not reloading checkpoint optimizer {name}.")
        #          for group_id, param_group in enumerate(self.optimizers[name].param_groups):
        #              if 'num_updates' not in param_group:
        #                  logger.warning(f"No 'num_updates' for optimizer {name}.")
        #                  continue
        #              logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer {name}.")
        #              param_group['num_updates'] = data[f'{name}_optimizer']['param_groups'][group_id]['num_updates']
        #              param_group['lr'] = self.optimizers[name].get_lr_for_step(param_group['num_updates'])

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.n_pros_sents = data["n_pros_sents"]
        # self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.n_sentences % self.params.save_periodic < 61
        ):
            self.save_checkpoint("checkpoint", include_optimizers=True)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[metric]))
                self.save_checkpoint("best-%s" % metric, include_optimizers=False)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        self.save_checkpoint("checkpoint", include_optimizers=True)
        self.epoch += 1


class EncDecTrainer(Trainer):
    def __init__(self, params):
        # model / data / params
        # # model = build_model(params)
        self.params = params
        super().__init__(params)

    def mt_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.model.train()

        # generate batch
        if lang1 == lang2:
            (x1, len1, w1) = self.get_batch("ae", lang1)
            (x2, len2, w2) = (x1, len1, w1)
            (x1, len1) = self.add_noise(x1, len1)
        else:
            (x1, len1, w1), (x2, len2, w2) = self.get_batch("mt", lang1, lang2)
            # if lang2 == 'ab':
            #     w2 = (w2 * (params.cdr_weight - 1)) + 1

        # cuda
        # TODO: GPU
        if self.params.cuda:
            x1, len1, x2, len2 = to_cuda(x1, len1, x2, len2)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(self.params.amp == 1)
        ):
            # encode source sentence
            loss = self.model(
                x1=x1, len1=len1, lang1=lang1, x2=x2, len2=len2, lang2=lang2
            )
            self.stats[
                ("AE-%s-loss" % lang1)
                if lang1 == lang2
                else ("MT-%s-%s-loss" % (lang1, lang2))
            ].append(loss.item())
            loss = lambda_coeff * loss

        # optimize
        self.optimize(
            loss,
            ("AE-%s-loss" % lang1)
            if lang1 == lang2
            else ("MT-%s-%s-loss" % (lang1, lang2)),
        )

        # number of processed sentences / words
        if lang1 == lang2:
            self.n_pros_sents[("AE_%s" % lang1)] += x1.shape[1]
        self.n_sentences += params.batch_size
        self.stats["processed_s"] += len2.size(0)
        self.stats["processed_w"] += (len2 - 1).sum().item()

    def get_sampling_tempreture(self, step, init_temp=1, decay_steps=2000, steepness=5):
        sigmoid = 1 / (1 + np.exp(steepness * ((step - 3000) / decay_steps - 0.1)))
        return init_temp * sigmoid + 1e-3

    def bt_step(self, lang1, lang2, lang3, lambda_coeff, optimize):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params
        _model = self.model.module if params.multi_gpu else self.model

        # generate source batch
        x1, len1, w1 = self.get_batch("bt", lang1)

        # cuda
        if self.params.cuda:
            x1, len1 = to_cuda(x1, len1)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(self.params.amp == 1)
        ):
            # generate a translation
            with torch.no_grad():
                # evaluation mode
                self.model.eval()

                x2, len2 = _model.generate(
                    x1,
                    len1,
                    lang1,
                    lang2,
                    temperature=self.get_sampling_tempreture(step=self.n_total_iter),
                )

                # training mode
                self.model.train()
        try:
            # encode generate sentence
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(self.params.amp == 1)
            ):
                loss = self.model(
                    x1=x2.transpose(0, 1),
                    len1=len2,
                    lang1=lang2,
                    x2=x1,
                    len2=len1,
                    lang2=lang1,
                )

                # loss
                self.stats[("BT-%s-%s-%s-loss" % (lang1, lang2, lang3))].append(
                    loss.item()
                )
                loss = lambda_coeff * loss
            self.optimize(loss, ("BT-%s-%s-%s-loss" % (lang1, lang2, lang3)), optimize)

            # number of processed sentences / words
            self.n_sentences += params.batch_size
            self.n_pros_sents[("BT_%s" % lang1)] += x1.shape[1]
            self.stats["processed_s"] += len1.size(0)
            self.stats["processed_w"] += (len1 - 1).sum().item()

        except:
            print("!!!" * 50)
            print(params.node_id)
            print(params.local_rank)
            print(params.global_rank)
            print(lang1, lang2)
            print(x1.shape)
            print(x1)
            print(len1.shape)
            print(len1)
            print(x2.shape)
            print(x2)
            print(len2.shape)
            print(len2)
            print("!!!" * 50)

        # optimize
        # self.optimize(loss, ("BT-%s-%s-%s-loss" % (lang1, lang2, lang3)), optimize)

        # number of processed sentences / words
        # self.n_sentences += params.batch_size
        # self.n_pros_sents[("BT_%s" % lang1)] += x1.shape[1]
        # self.stats["processed_s"] += len1.size(0)
        # self.stats["processed_w"] += (len1 - 1).sum().item()
