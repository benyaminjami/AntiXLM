# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from subprocess import call
import os
import sys
import torch
import socket
import signal
import subprocess


logger = getLogger()


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ['SLURM_PROCID'])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
        os.system('/opt/slurm/bin/scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    logger.warning("Signal handler installed.")


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm

    logger.info("SLURM job: %s" % str(params.is_slurm_job))
    
    
    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            logger.info(PREFIX + "%s: %s" % (name, str(value)))
            print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        # params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # params.local_rank = int(os.environ['RANK'])
        params.global_rank = int(os.environ['SLURM_PROCID'])
        # params.global_rank = int(os.environ['RANK'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ.get('WORLD_SIZE', 1))
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['/opt/slurm/bin/scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        logger.info(PREFIX + "Master address: %s" % params.master_addr)
        logger.info(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        # os.environ['MASTER_ADDR'] = params.master_addr
        # os.environ['MASTER_PORT'] = str(params.master_port)
        # os.environ['WORLD_SIZE'] = str(params.world_size)
        # os.environ['RANK'] = str(params.global_rank)
        params.master_addr = os.environ.get('MASTER_ADDR', params.master_addr)
        params.master_port = int(os.environ.get('MASTER_PORT', params.master_port))
        params.global_rank = int(os.environ.get('RANK', 0))

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1
    
    os.environ["NCCL_DEBUG"] = "INFO"

    if params.cuda:
        torch.cuda.set_device(params.local_rank)
    # summary
    PREFIX = "%i - " % params.global_rank
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "Local rank     : %i" % torch.cuda.current_device())
    logger.info(PREFIX + "Global rank    : %i" % params.global_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "GPUs per node-T: %i" % torch.cuda.device_count())
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())
    logger.info(PREFIX + "GPU            : %s" % torch.cuda.device(torch.cuda.current_device()))
    
    print(PREFIX + "Number of nodes: %i" % params.n_nodes)
    print(PREFIX + "Node ID        : %i" % params.node_id)
    print(PREFIX + "Local rank     : %i" % params.local_rank)
    print(PREFIX + "Local rank     : %i" % torch.cuda.current_device())
    print(PREFIX + "Global rank    : %i" % params.global_rank)
    print(PREFIX + "World size     : %i" % params.world_size)
    print(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    print(PREFIX + "GPUs per node-T: %i" % torch.cuda.device_count())
    print(PREFIX + "Master         : %s" % str(params.is_master))
    print(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())
    print(PREFIX + "GPU            : %s" % torch.cuda.device(torch.cuda.current_device()))
    # set GPU device
    

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    # assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node
    
    # initialize multi-GPU
    if params.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        logger.info("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            # init_method='env://',
            backend='nccl'
        )
        logger.info(PREFIX + "Initilized   ")