import sys

sys.path.append('../code')
import argparse
import torch

import os
from training.DPRecon_train import DPReconTrainRunner
import datetime

import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    #parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]') 
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel') # this is not required in torch 2.0
    parser.add_argument("--ft_folder", type=str, default=None, help='If set, finetune model from the given folder path')
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--none_wandb", default=False, action="store_true", help="If set, do not use wandb")
    parser.add_argument('--prior_yaml', type=str, required=True)
    parser.add_argument("--force_init_visgrid", default=False, action="store_true", help="If set, force to initialize visgrid from scratch.")

    opt = parser.parse_args()

    '''
    # if using GPUtil
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    '''
    # gpu = opt.local_rank
    gpu = int(os.environ["LOCAL_RANK"])

    seed_everything(42)

    # set distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # print(opt.local_rank)
    print('gpu: ', gpu)
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(1, 1800))
    torch.distributed.barrier()


    trainrunner = DPReconTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    scan_id=opt.scan_id,
                                    do_vis=not opt.cancel_vis,
                                    ft_folder = opt.ft_folder,
                                    description=opt.description,
                                    use_wandb=not opt.none_wandb,
                                    prior_yaml = opt.prior_yaml,
                                    force_init_visgrid = opt.force_init_visgrid
                                    )

    trainrunner.run()
