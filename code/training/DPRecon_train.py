import imp
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import json
import wandb
import trimesh
import random
from natsort import natsorted

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.general import get_time
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth
from datasets.scene_dataset import rot_cameras_along_x, rot_cameras_along_y
import matplotlib.pyplot as matplt

from model.VisField import VisGrid
from utils.get_prior_bbox_withvis import get_prior_bbox_withvis, fix_object_prior_bbox
import time
import shutil
from glob import glob

from threestudio.models.mesh import Mesh
from torch.utils.tensorboard import SummaryWriter


class DPReconTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.description = kwargs['description']
        self.use_wandb = kwargs['use_wandb']
        self.ft_folder = kwargs['ft_folder']
        self.prior_yaml = kwargs['prior_yaml']
        self.force_init_visgrid = kwargs['force_init_visgrid']

        self.use_extra_distillation_views = self.conf.get_bool('train.use_extra_distillation_views', default=False)
        self.extra_batch_size = self.conf.get_int('train.extra_batch_size', default=20)

        self.use_bg_inpainting = self.conf.get_bool('train.use_bg_inpainting', default=True)
        self.inpaint_repeat_times = self.conf.get_int('train.inpaint_repeat_times', default=10)

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)

        self.finetune_folder = kwargs['ft_folder'] if kwargs['ft_folder'] is not None else None
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        if self.GPU_INDEX == 0:
            utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
            self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
            utils.mkdir_ifnotexists(self.expdir)
            if self.description == "":
                self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            else:
                self.timestamp = f'{self.description}' + '_{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
            utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

            self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
            utils.mkdir_ifnotexists(self.plots_dir)

            # create checkpoints dirs
            self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
            utils.mkdir_ifnotexists(self.checkpoints_path)
            self.model_params_subdir = "ModelParameters"
            self.optimizer_params_subdir = "OptimizerParameters"
            self.scheduler_params_subdir = "SchedulerParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

            os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        # if (not self.GPU_INDEX == 'ignore'):
        #     os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('[INFO]: shell command : {0}'.format(' '.join(sys.argv)))

        print('[INFO]: Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)

        self.max_total_iters = self.conf.get_int('train.max_total_iters', default=200000)
        self.ds_len = len(self.train_dataset)
        print('Finish loading data. Data-set size: {0}'.format(self.ds_len))

        self.nepochs = int(self.max_total_iters / self.ds_len)
        print('RUNNING FOR {0}'.format(self.nepochs))

        if len(self.train_dataset.label_mapping) > 0:
            # a hack way to let network know how many categories, so don't need to manually set in config file
            self.conf['model']['implicit_network']['d_out'] = len(self.train_dataset.label_mapping)
            print('RUNNING FOR {0} CLASSES'.format(len(self.train_dataset.label_mapping)))

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8,
                                                            pin_memory=True)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )
        
        self.n_sem = self.conf.get_int('model.implicit_network.d_out')
        assert self.n_sem == len(self.train_dataset.label_mapping)

        data_type = self.conf.get_string('dataset.data_dir')
        self.data_type = data_type
        prompt_type = self.prior_yaml.split('/')[-1].split('.')[0]
        self.training_mode = prompt_type                # geometry = recon + geo prior, texture = color prior
        prompt_path = os.path.join(self.train_dataset.instance_dir, f'{data_type}_scan{scan_id}_prompt_{prompt_type}.json')
        self.fix_location_json_path = os.path.join(self.train_dataset.instance_dir, f'{data_type}_scan{scan_id}_location_modify.json')
        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model, prior_yaml=self.prior_yaml, plots_dir=self.plots_dir, ft_folder=self.ft_folder, obj_num=self.n_sem, prompt_path=prompt_path)

        self.use_visgrid = True
        self.init_visgrid = True
        self.init_color_mesh = True

        self.visgrid_optim_iter = self.conf.get_int('train.visgrid_optim_iter')
        if self.use_visgrid:
            self.visgrid = VisGrid(fill_data=0.0)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # The MLP and hash grid should have different learning rates
        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        
        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
                {'name': 'visgrid', 'params': list(self.visgrid.grid_parameters()),
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'mesh_color_net', 'params': list(self.model.mesh_color_network.parameters()),
                    'lr': self.lr * self.lr_factor_for_grid}
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.nepochs * len(self.train_dataset)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.GPU_INDEX], broadcast_buffers=False, find_unused_parameters=True)
        
        self.do_vis = kwargs['do_vis']

        self.start_epoch = 0
        # Loading a pretrained model for finetuning, the model path can be provided by self.finetune_folder
        if is_continue or self.finetune_folder is not None:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints') if self.finetune_folder is None\
             else os.path.join(self.finetune_folder, 'checkpoints')

            print('[INFO]: Loading pretrained model from {}'.format(old_checkpnts_dir))
            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            if "visgrid_state_dict" in saved_model_state and not self.force_init_visgrid:                       # if force_init_visgrid is True, we will not load visgrid
                self.visgrid.load_state_dict(saved_model_state["visgrid_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            
            visgrid_op_exist = False
            for op_groups in data['optimizer_state_dict']['param_groups']:
                if 'visgrid' in op_groups['name']:
                    visgrid_op_exist = True
                    break
            if not visgrid_op_exist:
                copy_groups = data['optimizer_state_dict']['param_groups'][0].copy()
                copy_groups['name'] = 'visgrid'
                copy_groups['params'] = list(self.visgrid.grid_parameters())
                data['optimizer_state_dict']['param_groups'].append(copy_groups)

            mesh_color_net_op_exist = False
            for op_groups in data['optimizer_state_dict']['param_groups']:
                if 'mesh_color_net' in op_groups['name']:
                    mesh_color_net_op_exist = True
                    break
            if not mesh_color_net_op_exist:
                copy_groups = data['optimizer_state_dict']['param_groups'][0].copy()
                copy_groups['name'] = 'mesh_color_net'
                copy_groups['params'] = list(self.model.module.mesh_color_network.parameters())
                copy_groups['lr'] = 0.01
                data['optimizer_state_dict']['param_groups'].append(copy_groups)

            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()
        
        self.add_objectvio_iter = self.conf.get_int('train.add_objectvio_iter', default=10000)
        self.add_prior_iter = self.conf.get_int('train.add_prior_iter', default=60000)
        self.add_color_prior_iter = self.conf.get_int('train.add_color_prior_iter', default=90000)

        color_max_iter = self.conf.get_int('train.color_max_iter', default=1000)
        self.color_epoch_num = int(color_max_iter / self.ds_len)
        print(f'color_epoch_num: {self.color_epoch_num}')
        self.scene_rgb_loss_weight = self.conf.get_float('train.scene_rgb_loss_weight', default=100.0)
        self.bg_pano_rgb_loss_weight = self.conf.get_float('train.bg_pano_rgb_loss_weight', default=10000.0)
        self.sds_rgb_loss_weight = self.conf.get_float('train.sds_rgb_loss_weight', default=0.0001)

        self.iter_step = self.start_epoch * len(self.train_dataset)
        if self.iter_step >= self.add_prior_iter and not self.force_init_visgrid:
            self.model.module.begin_prior = True
            self.init_visgrid = False

            # continue training, copy prior bbox to current folder
            continue_prior_bbox_path = os.path.join(self.finetune_folder, 'plots', 'prior_bbox')
            if not os.path.exists(continue_prior_bbox_path):
                ValueError(f'continue_prior_bbox_path not exists: {continue_prior_bbox_path}')
            dst_prior_bbox_path = os.path.join(self.plots_dir, 'prior_bbox')
            shutil.copytree(continue_prior_bbox_path, dst_prior_bbox_path)


    def save_checkpoints(self, epoch):
        if self.use_visgrid:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict(), "visgrid_state_dict": self.visgrid.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict(), "visgrid_state_dict": self.visgrid.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        else:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def run(self):

        print("training...")
        self.use_tensorboard = not self.use_wandb           # use tensorboard as default
        if self.GPU_INDEX == 0 :

            if self.use_tensorboard and self.use_wandb:
                print('Please choose either tensorboard or wandb to visualize the training process.')
                self.use_wandb = False
                print('Use tensorboard to visualize the training process.')

            if self.use_wandb:
                infos = json.loads(json.dumps(self.conf))
                wandb.init(
                    config=infos,
                    project=self.conf.get_string('wandb.project_name'),
                    name=self.timestamp,
                    # notes='description',
                    # group='group1 --> tag',
                )

                # # visiualize gradient
                # wandb.watch(self.model, self.optimizer)

            if self.use_tensorboard:
                self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        print(f'Start epoch: {self.start_epoch}, iter_step: {self.iter_step}')
        for epoch in range(self.start_epoch, self.nepochs + 1):

            if self.GPU_INDEX == 0 and epoch % self.checkpoint_freq == 0 and epoch != 0:
                self.save_checkpoints(epoch)

            if self.GPU_INDEX == 0 and self.do_vis and epoch % self.plot_freq == 0 and epoch != 0 and self.training_mode == 'geometry':
                self.model.eval()
                self.visgrid.eval()

                self.train_dataset.change_sampling_idx(-1)

                rendering_list = [i for i in range(self.train_dataset.n_images)]
                render_view_num = 3
                if len(rendering_list) > render_view_num:
                    # random select 10 views
                    random.shuffle(rendering_list)
                    rendering_list = rendering_list[:render_view_num]
                for img_idx in rendering_list:
                    indices, model_input, ground_truth = self.train_dataset[img_idx]

                    indices = torch.tensor([indices]).cuda()

                    model_input["intrinsics"] = model_input["intrinsics"].unsqueeze(0).cuda()
                    model_input["uv"] = model_input["uv"].unsqueeze(0).cuda()
                    model_input['pose'] = model_input['pose'].unsqueeze(0).cuda()

                    ground_truth['rgb'] = ground_truth['rgb'].unsqueeze(0).cuda()
                    ground_truth['normal'] = ground_truth['normal'].unsqueeze(0).cuda()
                    ground_truth['depth'] = ground_truth['depth'].unsqueeze(0).cuda()
                    ground_truth['segs'] = ground_truth['segs'].unsqueeze(0).cuda()
                    ground_truth['mask'] = ground_truth['mask'].unsqueeze(0).cuda()
                    
                    split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                    res = []
                    for s in tqdm(split):
                        out = self.model(s, indices)

                        points_flat = out['points_flat']
                        weights = out['weights']
                        max_trans = self.visgrid(points_flat)          # [ray_num*N_samples, 1]
                        ray_num = weights.shape[0]
                        N_samples = weights.shape[1]
                        with torch.no_grad():
                            max_trans = max_trans.reshape(-1, N_samples, 1)
                        vis_values = torch.sum(weights.unsqueeze(-1) * max_trans, 1)

                        d = {'rgb_values': out['rgb_values'].detach(),
                            'normal_map': out['normal_map'].detach(),
                            'depth_values': out['depth_values'].detach(),
                            'vis_values': vis_values.detach(),}
                        if 'rgb_un_values' in out:
                            d['rgb_un_values'] = out['rgb_un_values'].detach()
                        if 'semantic_values' in out:
                            d['semantic_values'] = torch.argmax(out['semantic_values'].detach(),dim=1)
                        res.append(d)

                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                    plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['segs'])

                    obj_bbox_dict = None
                    if img_idx == rendering_list[-1]:
                        plot_mesh = True

                        if os.path.exists(os.path.join(self.plots_dir, 'prior_bbox')):          # use prior bbox first
                            obj_bbox_root_path = os.path.join(self.plots_dir, 'prior_bbox')
                        elif os.path.exists(os.path.join(self.plots_dir, 'bbox')):              # use object bbox
                            obj_bbox_root_path = os.path.join(self.plots_dir, 'bbox')
                        else:
                            obj_bbox_root_path = None
                        
                        if obj_bbox_root_path != None:        # use object bbox
                            obj_bbox_dict = {}
                            obj_list = os.listdir(obj_bbox_root_path)
                            obj_list = [obj for obj in obj_list if obj.endswith('.json')]
                            for obj in obj_list:
                                obj_idx = int((obj.split('.')[0]).split('_')[1])
                                with open(os.path.join(obj_bbox_root_path, obj), 'r') as f:
                                    bbox = json.load(f)
                                obj_bbox_dict[obj_idx] = bbox
                    else:
                        plot_mesh = False
                    
                    plt.plot(self.model.module.implicit_network,
                            indices,
                            plot_data,
                            self.plots_dir,
                            epoch,
                            self.iter_step,         # iter
                            self.img_res,
                            **self.plot_conf,
                            plot_mesh=plot_mesh,
                            obj_bbox_dict=obj_bbox_dict
                            )

                self.model.train()

            if self.iter_step >= self.add_prior_iter and self.init_visgrid and epoch % self.plot_freq == 0:
                self.init_visgrid = False                                               # only init visgrid for the first time
                print('********** begin visgrid initialization **********')

                t1 = time.time()

                self.model.eval()
                self.visgrid.train()
                self.train_dataset.change_sampling_idx(-1)

                max_visgrid_list = []
                mean_visgrid_list = []
                min_visgrid_list = []
                loss_list = []

                total_init_iters = self.visgrid_optim_iter
                print(f'********** begin visgrid initialization, total iteration: {total_init_iters} **********')
                for init_idx in range(total_init_iters):                          # optimize visgrid for total_init_iters iterations

                    rendering_list = [i for i in range(self.train_dataset.n_images)]
                    total_loss = 0
                    for img_idx in rendering_list:
                        indices, model_input, ground_truth = self.train_dataset[img_idx]
                        indices = torch.tensor([indices]).cuda()

                        model_input["intrinsics"] = model_input["intrinsics"].unsqueeze(0).cuda()
                        model_input["uv"] = model_input["uv"].unsqueeze(0).cuda()
                        model_input['pose'] = model_input['pose'].unsqueeze(0).cuda()

                        self.optimizer.zero_grad()

                        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                        res = []
                        loss = 0
                        for s in tqdm(split):
                            out = self.model(s, indices)
                            
                            points_flat = out['points_flat']
                            transmittance = out['transmittance']
                            max_trans = self.visgrid(points_flat)          # [ray_num*N_samples, 1]
                            visgrid_loss = self.visgrid.get_loss(max_trans, transmittance)
                            loss += visgrid_loss

                        loss.backward()

                        # calculate gradient norm
                        total_norm = 0
                        parameters = [p for p in self.visgrid.parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                        self.optimizer.step()
                        self.scheduler.step()

                        max_visgrid = torch.max(self.visgrid.visgrid.data)
                        mean_visgrid = torch.mean(self.visgrid.visgrid.data)
                        min_visgrid = torch.min(self.visgrid.visgrid.data)
                        max_visgrid_list.append(max_visgrid.item())
                        mean_visgrid_list.append(mean_visgrid.item())
                        min_visgrid_list.append(min_visgrid.item())
                        loss_list.append(loss.item())
                        total_loss += loss.item()
                        print(f'visgrid loss: {loss.item()}, total_norm: {total_norm}, max_visgrid: {max_visgrid}, mean_visgrid: {mean_visgrid}')

                    avg_loss = total_loss / len(rendering_list)
                    print(f'********** average loss: {avg_loss} **********')

                t2 = time.time()
                print(f'********** finish visgrid initialization, running time {t2 - t1}s for {total_init_iters} iteration, avg loss {avg_loss} **********')
                
                # save visgrid data
                max_visgrid_list = np.array(max_visgrid_list)
                mean_visgrid_list = np.array(mean_visgrid_list)
                min_visgrid_list = np.array(min_visgrid_list)
                loss_list = np.array(loss_list)
                max_visgrid_path = os.path.join(self.plots_dir, 'max_visgrid.npy')
                mean_visgrid_path = os.path.join(self.plots_dir, 'mean_visgrid.npy')
                min_visgrid_path = os.path.join(self.plots_dir, 'min_visgrid.npy')
                loss_path = os.path.join(self.plots_dir, 'loss.npy')
                np.save(max_visgrid_path, max_visgrid_list)
                np.save(mean_visgrid_path, mean_visgrid_list)
                np.save(min_visgrid_path, min_visgrid_list)
                np.save(loss_path, loss_list)

                # plot loss curve
                matplt.figure()
                matplt.plot(loss_list, label='loss')
                matplt.xlabel('iteration')
                matplt.ylabel('loss')
                matplt.title('visgrid loss curve')
                matplt.legend()
                matplt.savefig(os.path.join(self.plots_dir, 'visgrid_loss_curve.png'))
                matplt.close()

                # plot visgrid data
                matplt.figure()
                matplt.plot(max_visgrid_list, label='max_visgrid')
                matplt.plot(mean_visgrid_list, label='mean_visgrid')
                matplt.plot(min_visgrid_list, label='min_visgrid')
                matplt.xlabel('iteration')
                matplt.ylabel(' value')
                matplt.title('visgrid value curve')
                matplt.legend()
                matplt.savefig(os.path.join(self.plots_dir, 'visgrid_value_curve.png'))
                matplt.close()

                self.model.module.begin_prior = True
                print('********** begin add object prior **********')

                if self.GPU_INDEX == 0:
                    self.save_checkpoints(epoch)

                self.model.train()
                self.visgrid.eval()

                # set prior bbox
                vis_thresh = 0.02 * self.ds_len                                     # more training views, higher vis_thresh
                print(f'use vis_thresh: {vis_thresh}')
                _ = get_prior_bbox_withvis(self.visgrid, self.plots_dir, epoch, vis_thresh)
                fix_object_prior_bbox(self.plots_dir, self.fix_location_json_path)


            # end geometry training, infer all views and get background panorama map
            if self.iter_step >= self.add_color_prior_iter and self.training_mode == 'geometry':

                self.model.eval()
                self.visgrid.eval()

                print('********** finish geometry optimization, generate bg panorama **********')
                from utils.pano_camera_tools import capture_pano_depth
                from PIL import Image

                # bg_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_0.ply')
                bg_mesh_path = natsorted(glob(os.path.join(self.plots_dir, f'surface_*_0.ply')))[-1]
                if not os.path.exists(bg_mesh_path):
                    # ft_bg_mesh_path = os.path.join(self.ft_folder, 'plots', f'surface_{epoch}_0.ply')
                    ft_bg_mesh_path = natsorted(glob(os.path.join(self.ft_folder, 'plots', f'surface_*_0.ply')))[-1]
                    shutil.copyfile(ft_bg_mesh_path, bg_mesh_path)

                mesh = trimesh.load(bg_mesh_path)
                # calculate the centroid
                vertices = mesh.vertices
                x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
                y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
                z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                z_center = (z_min + z_max) / 2
                print(f'mesh center: {x_center}, {y_center}, {z_center}')
                centroid = np.array([x_center, y_center, z_center])
                mesh.vertices = mesh.vertices - centroid
                # rot -90 along x axis
                mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))

                img_h = 1024
                _, disp, _, points, pixel_ray = capture_pano_depth(mesh, img_h, 0.0, [0.0, 0.0, 0.0])

                # points rot 90 along x axis
                x_rot_mat = np.array([
                    [1, 0, 0], 
                    [0, 0, 1], 
                    [0, -1, 0]
                ])
                points = points @ x_rot_mat
                points = points + centroid

                # query the model
                points = torch.from_numpy(points).float().cuda()
                max_trans = self.visgrid(points).detach().cpu().numpy().squeeze()
                vis_map = np.zeros((img_h, 2*img_h))
                vis_map[pixel_ray[:, 0], pixel_ray[:, 1]] = max_trans
                vis_mask = (vis_map < 0.2).astype('uint8')                                    # True means need to be inpainted

                points_split = points.shape[0] // 32
                total_rgb = []
                cam_center = torch.tensor([0.0, 0.0, 0.0]).float().cuda()
                for i in range(0, points.shape[0], points_split):
                    points_batch = points[i:i+points_split]
                    _, feature_vectors, gradients, _, _ = self.model.module.implicit_network.get_outputs(points_batch, beta=None)
                    dirs_flat = points_batch - cam_center
                    dirs_flat = F.normalize(dirs_flat, dim=-1)          # [N, 3]
                    rgb_flat = self.model.module.rendering_network(points_batch, gradients, dirs_flat, feature_vectors, indices=0)
                    rgb_flat = rgb_flat.detach().cpu().numpy()
                    total_rgb.append(rgb_flat)
                total_rgb = np.concatenate(total_rgb, axis=0)
                rgb_map = np.zeros((img_h, 2*img_h, 3))
                rgb_map[pixel_ray[:, 0], pixel_ray[:, 1]] = total_rgb

                pano_save_root_path = os.path.join(self.plots_dir, 'sds_views', 'obj_0', 'bg_pano')
                os.makedirs(pano_save_root_path, exist_ok=True)
                disp_npy_path = os.path.join(pano_save_root_path, 'disp.npy')
                np.save(disp_npy_path, disp)
                disp_path = os.path.join(pano_save_root_path, 'disp.png')
                disp = (disp - disp.min()) / (disp.max() - disp.min())
                disp = Image.fromarray((disp*255).astype('uint8'))
                disp.save(disp_path)

                vis_mask_path = os.path.join(pano_save_root_path, 'vis_mask.png')
                vis_mask = Image.fromarray((vis_mask*255).astype('uint8'))
                vis_mask.save(vis_mask_path)

                rgb_map_path = os.path.join(pano_save_root_path, 'rgb_map.png')
                rgb_map = Image.fromarray((rgb_map*255).astype('uint8'))
                rgb_map.save(rgb_map_path)

                # save vis map
                vis_map_path = os.path.join(pano_save_root_path, 'vis_map.npy')
                np.save(vis_map_path, vis_map)
                vis_map_img_path = os.path.join(pano_save_root_path, 'vis_map.png')
                matplt.imsave(vis_map_img_path, vis_map, cmap='viridis')

                # save masked rgb map
                masked_rgb_map = np.array(rgb_map)
                vis_mask_map = np.array(vis_mask.convert("RGB"))
                masked_rgb_map[np.where(vis_mask_map == 255)] = 255
                masked_rgb_map = Image.fromarray(masked_rgb_map)
                masked_rgb_map_path = os.path.join(pano_save_root_path, 'masked_rgb_map.png')
                masked_rgb_map.save(masked_rgb_map_path)


                print('*********** filter bg artifacts by TSDF for better bg UV map ***********')
                save_bg_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_0_filtered.ply')
                input_scalemat_path = os.path.join(self.train_dataset.instance_dir, 'eval-novel-views', 'cameras.npz')
                tsdf_filter_cmd = f'python utils/tsdf_filter_util.py --input_mesh {bg_mesh_path} --save_mesh {save_bg_mesh_path} --input_scalemat {input_scalemat_path}'
                print(tsdf_filter_cmd)
                try:
                    os.system(tsdf_filter_cmd)
                    # replace bg_mesh_path with filtered mesh
                    ori_bg_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_0_ori.ply')
                    shutil.move(bg_mesh_path, ori_bg_mesh_path)
                    shutil.copyfile(save_bg_mesh_path, bg_mesh_path)
                    print(f'bg mesh filtered, save to {bg_mesh_path}')
                except:
                    print('[INFO]: bg tsdf filter failed, use original bg mesh')


                print('********** finish bg panorama generation, begin render color image **********')

                # infer nerf rgb, nerf mask
                infer_camera_files = os.path.join(self.train_dataset.instance_dir, 'eval-novel-views', 'cameras.npz')               # infer all 100 views
                infer_camera_dict = np.load(infer_camera_files)

                eval_json_path = os.path.join(self.train_dataset.instance_dir, 'eval-novel-views', 'eval-10views.json')             # 10 views for evaluation
                with open(eval_json_path, 'r') as f:
                    eval_view_list = json.load(f)
                nerf_rgb_root_path = os.path.join(self.plots_dir, 'eval_nerf_rgb')
                os.makedirs(nerf_rgb_root_path, exist_ok=True)
                nerf_mask_root_path = os.path.join(self.plots_dir, 'eval_nerf_mask')
                os.makedirs(nerf_mask_root_path, exist_ok=True)
                save_nerf_anchor_root_path = os.path.join(self.plots_dir, 'nerf_rgb_anchor_views')
                os.makedirs(save_nerf_anchor_root_path, exist_ok=True)

                infer_views_num = 10
                scale_mats = [infer_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(infer_views_num)]
                world_mats = [infer_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(infer_views_num)]
                infer_camera_pose_list = []
                cam_intrinsics = self.train_dataset.intrinsics_all[0]       # use the same intrinsics for all views
                
                for scale_mat, world_mat in zip(scale_mats, world_mats):

                    P = world_mat @ scale_mat
                    P = P[:3, :4]
                    intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

                    if self.train_dataset.need_fix_y:
                        pose = rot_cameras_along_y(pose, self.train_dataset.fix_y_deg)              # need to fix the camera pose along y axis for replica scan3
                    if self.train_dataset.need_fix_x:
                        pose = rot_cameras_along_x(pose, self.train_dataset.fix_x_deg)              # need to fix the camera pose along x axis for youtube scan1 and scan2

                    pose = torch.from_numpy(pose).float()
                    infer_camera_pose_list.append(pose)
                infer_camera_pose = torch.stack(infer_camera_pose_list, dim=0)
                infer_camera_pose = infer_camera_pose.reshape(-1, 4, 4)

                if self.use_extra_distillation_views:                       # infer for distill NeRF color mlp to nvdiffrast mlp
                    
                    print('****************** Add extra distillation views for color mlp distillation ******************')
                    
                    input_pose_all = self.train_dataset.pose_all            # use training view camera centers
                    self.model.module.prior.camera_module.batch_size = self.extra_batch_size
                    anchor_camera_pose_list = []
                    for pose in input_pose_all:
                        cam_center = pose[:3, 3]
                        cam_batch = self.model.module.prior.camera_module.get_scene_camera(camera_center=cam_center, elevation_range=[-30, 10])
                        c2w = cam_batch['c2w']

                        # colmap to opengl
                        c2w[:, :3, 1] *= -1
                        c2w[:, :3, 2] *= -1
                        
                        anchor_camera_pose_list.append(c2w)
                    anchor_camera_pose = torch.stack(anchor_camera_pose_list, dim=0)
                    anchor_camera_pose = anchor_camera_pose.reshape(-1, 4, 4)

                    # cat anchor camera pose to infer camera pose
                    infer_camera_pose = torch.cat([infer_camera_pose, anchor_camera_pose], dim=0)

                # save anchor camera pose and intrinsics for distillation
                anchor_camera_pose_save_path = os.path.join(save_nerf_anchor_root_path, 'anchor_camera_pose.npy')
                np.save(anchor_camera_pose_save_path, infer_camera_pose.numpy())
                anchor_intrinsics_save_path = os.path.join(save_nerf_anchor_root_path, 'anchor_intrinsics.npy')
                np.save(anchor_intrinsics_save_path, cam_intrinsics.numpy())

                total_anchor_num = infer_camera_pose.shape[0]
                print(f'********** infer camera pose number: {total_anchor_num} **********')
                for idx in range(total_anchor_num):
                    model_input = {}
                    model_input['pose'] = infer_camera_pose[idx].unsqueeze(0).cuda().float()
                    model_input['intrinsics'] = cam_intrinsics.unsqueeze(0).cuda().float()

                    uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
                    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
                    uv = uv.reshape(2, -1).transpose(1, 0)
                    model_input["uv"] = uv.unsqueeze(0).cuda()

                    indices = torch.tensor([idx]).cuda()
                    split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                    res = []
                    for s in tqdm(split):
                        # if render_bg:
                        #     out = self.model(s, indices, render_obj_idx=0)
                        # else:
                        out = self.model(s, indices)

                        points_flat = out['points_flat']
                        weights = out['weights']
                        max_trans = self.visgrid(points_flat)          # [ray_num*N_samples, 1]
                        ray_num = weights.shape[0]
                        N_samples = weights.shape[1]
                        max_trans = max_trans.reshape(-1, N_samples, 1)
                        vis_values = torch.sum(weights.unsqueeze(-1) * max_trans, 1)

                        d = {'rgb_values': out['rgb_values'].detach(),
                            'normal_map': out['normal_map'].detach(),
                            'depth_values': out['depth_values'].detach(),
                            'vis_values': vis_values.detach(),}
                        if 'rgb_un_values' in out:
                            d['rgb_un_values'] = out['rgb_un_values'].detach()
                        if 'semantic_values' in out:
                            d['semantic_values'] = torch.argmax(out['semantic_values'].detach(),dim=1)
                        res.append(d)

                    batch_size = 1
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                    num_samples = model_outputs['rgb_values'].shape[0]

                    rgb_gt = torch.zeros(batch_size, num_samples, 3).cuda()
                    normal_gt = torch.zeros(batch_size, num_samples, 3).cuda()
                    depth_gt = model_outputs['depth_values'].unsqueeze(0)               # get_plot_data will scale the pred depth to match the gt depth, so we use gt depth here
                    seg_gt = torch.zeros(batch_size, num_samples, 1).cuda()

                    plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], rgb_gt, normal_gt, depth_gt, seg_gt)

                    nerf_mask = model_outputs['semantic_values'].reshape(self.img_res[0], self.img_res[1]).detach().cpu().numpy()
                    nerf_rgb = model_outputs['rgb_values'].reshape(self.img_res[0], self.img_res[1], 3).detach().cpu().numpy()
                    nerf_rgb = (nerf_rgb * 255).astype(np.uint8)
                    nerf_rgb = Image.fromarray(nerf_rgb)

                    # save anchor nerf rgb
                    anchor_save_path = os.path.join(save_nerf_anchor_root_path, f'nerf_rgb_{idx}.png')
                    nerf_rgb.save(anchor_save_path)

                    if idx in eval_view_list:
                        # save eval nerf mask
                        nerf_mask_npy_path = os.path.join(nerf_mask_root_path, f'nerf_mask_{epoch}_{idx}.npy')
                        np.save(nerf_mask_npy_path, nerf_mask)
                        nerf_mask = Image.fromarray((nerf_mask * 255).astype(np.uint8))
                        nerf_mask_path = os.path.join(nerf_mask_root_path, f'nerf_mask_{epoch}_{idx}.png')
                        nerf_mask.save(nerf_mask_path)
                    
                        # save eval nerf rgb
                        nerf_rgb_path = os.path.join(nerf_rgb_root_path, f'nerf_rgb_{epoch}_{idx}.png')
                        nerf_rgb.save(nerf_rgb_path)

                print('********** finish render color image, begin inpaint bg panorama **********')

                save_bg_inpaint_path = os.path.join(pano_save_root_path, 'bg_inpaint.png')
                if self.use_bg_inpainting:
                    # inpaint bg panorama
                    print('********** begin inpaint bg panorama **********')
                    inpaint_save_path = os.path.join(pano_save_root_path, 'inpaint')
                    os.makedirs(inpaint_save_path, exist_ok=True)
                    inpaint_cmd = f'python ControlNetInpaint/inpaint_pano.py --root_dir {pano_save_root_path} --save_dir {inpaint_save_path} --inpaint_repeat_times {self.inpaint_repeat_times}'
                    print('inpaint_cmd: ', inpaint_cmd)
                    # os.system(inpaint_cmd)                        # NOTE: may occur OOM error when using os.system due to not freed memory
                    # inpaint_result_list = os.listdir(inpaint_save_path)
                    # inpaint_result_list = [result for result in inpaint_result_list if result.endswith('.png')]
                    # random_inpaint_result_name = random.choice(inpaint_result_list)
                    # random_inpaint_result_path = os.path.join(inpaint_save_path, random_inpaint_result_name)
                    # shutil.copy(random_inpaint_result_path, save_bg_inpaint_path)
                    # print('[INFO]: finish inpaint bg panorama. A result has been randomly selected, but manually choosing the best one from all results would yield better quality.')
                    # print(f'[INFO]: all the inpainting results are saved at {inpaint_save_path}, please choose the best one to replace {save_bg_inpaint_path}')

                    # NOTE: use os.execvp to avoid OOM error, but os.execvp will not return to the current process
                    # save ckpt before exit current process
                    if self.GPU_INDEX == 0:
                        self.save_checkpoints(epoch)

                    if self.use_wandb:
                        wandb.finish()

                    print('[INFO] training over, exit current process and begin inpaint bg panorama')
                    execvp_inpaint_cmd = [
                        'python',
                        'ControlNetInpaint/inpaint_pano.py',
                        '--root_dir',
                        pano_save_root_path,
                        '--save_dir',
                        inpaint_save_path,
                        '--inpaint_repeat_times',
                        str(self.inpaint_repeat_times)                  # need to convert to str
                    ]
                    os.execvp('python', execvp_inpaint_cmd)
                else:
                    shutil.copy(rgb_map_path, save_bg_inpaint_path)
                    print('[INFO]: WARNING: not use bg inpainting, directly use original bg panorama')

                print('********** finish inpaint bg panorama, end decompositional reconstruction and add geometry prior stage, exit **********')
                break

            if self.iter_step >= self.add_color_prior_iter and self.init_color_mesh:

                if self.ft_folder is not None:
                    src_pano_path = os.path.join(self.ft_folder, 'plots', 'sds_views', 'obj_0', 'bg_pano')
                    dst_pano_path = os.path.join(self.plots_dir, 'sds_views', 'obj_0', 'bg_pano')
                    shutil.copytree(src_pano_path, dst_pano_path)

                    # copy init mesh
                    src_mesh_list = os.listdir(os.path.join(self.ft_folder, 'plots'))
                    # mesh_str = f'surface_{epoch}_'
                    src_mesh_list = [mesh for mesh in src_mesh_list if mesh.endswith('.ply')]
                    for mesh in src_mesh_list:
                        src_mesh_path = os.path.join(self.ft_folder, 'plots', mesh)
                        dst_mesh_path = os.path.join(self.plots_dir, mesh)
                        shutil.copy(src_mesh_path, dst_mesh_path)

                    # copy eval nerf rgb and mask
                    src_nerf_rgb_path = os.path.join(self.ft_folder, 'plots', 'eval_nerf_rgb')
                    dst_nerf_rgb_path = os.path.join(self.plots_dir, 'eval_nerf_rgb')
                    shutil.copytree(src_nerf_rgb_path, dst_nerf_rgb_path)

                    src_nerf_mask_path = os.path.join(self.ft_folder, 'plots', 'eval_nerf_mask')
                    dst_nerf_mask_path = os.path.join(self.plots_dir, 'eval_nerf_mask')
                    shutil.copytree(src_nerf_mask_path, dst_nerf_mask_path)

                else:
                    raise ValueError('no ft_folder found')

                self.init_color_mesh = False
                self.model.module.begin_color_prior = True
                self.model.module.prior.begin_color_prior = True

                print('********** begin color prior, freeze geometry network **********')
                # detach all parameters except the color network parameters
                # 1. encoding parameters
                for param in self.model.module.implicit_network.grid_parameters():
                    param.requires_grad = False
                # 2. implicit network parameters
                for param in self.model.module.implicit_network.mlp_parameters():
                    param.requires_grad = False
                # 3. density parameters
                for param in self.model.module.density.parameters():
                    param.requires_grad = False
                # 4. visgrid parameters
                for param in self.visgrid.grid_parameters():
                    param.requires_grad = False

                # set color anchor views
                src_bg_anchor_views = os.path.join(self.ft_folder, 'plots', 'sds_views', 'obj_0', 'bg_pano_anchor')
                if os.path.exists(src_bg_anchor_views):
                    dst_bg_anchor_views = os.path.join(self.plots_dir, 'sds_views', 'obj_0', 'bg_pano_anchor')
                    shutil.copytree(src_bg_anchor_views, dst_bg_anchor_views)
                else:
                    self.model.module.bg_pano_anchor_views()

                # set color mesh render module
                self.model.module.prior.init_color_render_module(mesh_color_network=self.model.module.mesh_color_network)
                self.model.module.mesh_color_network.visgrid = self.visgrid
                print('use color mesh render module')

                # set color mesh
                prior_bbox_root_path = os.path.join(self.plots_dir, 'prior_bbox')
                for obj_idx in self.model.module.prior_obj_idx_list:         # include bg and obj
                    # obj_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_{obj_idx}.ply')
                    obj_mesh_paths = natsorted(glob(os.path.join(self.plots_dir, f'surface_*_{obj_idx}.ply')))
                    if len(obj_mesh_paths) == 0:
                        continue
                    obj_mesh_path = obj_mesh_paths[-1]
                    if obj_idx == 0:
                        prior_bbox_path = None
                    else:
                        prior_bbox_path = os.path.join(prior_bbox_root_path, f'bbox_{obj_idx}.json')
                    self.model.module.prior.init_color_mesh([obj_mesh_path], obj_idx, prior_bbox_path)
                        
                obj_mesh_path_list = []
                for i in range(self.n_sem):
                    # i_mesh_path = os.path.join(self.plots_dir, f'surface_{epoch}_{i}.ply')
                    i_mesh_paths = natsorted(glob(os.path.join(self.plots_dir, f'surface_*_{i}.ply')))
                    if len(i_mesh_paths) == 0:
                        continue
                    i_mesh_path = i_mesh_paths[-1]
                    obj_mesh_path_list.append(i_mesh_path)
                prior_bbox_path = None
                self.model.module.prior.init_color_mesh(obj_mesh_path_list, 1000, prior_bbox_path)   # add total scene


            if self.model.module.begin_color_prior:                     # begin color optimization stage

                from PIL import Image

                src_nerf_anchor_path = os.path.join(self.ft_folder, 'plots', 'nerf_rgb_anchor_views')
                if os.path.exists(src_nerf_anchor_path):
                    dst_nerf_anchor_path = os.path.join(self.plots_dir, 'nerf_rgb_anchor_views')
                    shutil.copytree(src_nerf_anchor_path, dst_nerf_anchor_path)
                    print(f'load nerf anchor views from {src_nerf_anchor_path}')

                    save_nerf_anchor_root_path = dst_nerf_anchor_path
                    anchor_camera_pose_path = os.path.join(save_nerf_anchor_root_path, 'anchor_camera_pose.npy')
                    anchor_intrinsics_path = os.path.join(save_nerf_anchor_root_path, 'anchor_intrinsics.npy')
                    anchor_camera_pose = torch.from_numpy(np.load(anchor_camera_pose_path))
                    anchor_intrinsics = torch.from_numpy(np.load(anchor_intrinsics_path))
                    total_anchor_num = anchor_camera_pose.shape[0]
                else:
                    raise ValueError('no nerf anchor views found')

                print(f'********** begin color optimization, total {self.color_epoch_num} epochs, scene rgb loss weight {self.scene_rgb_loss_weight}, bg pano rgb loss weight {self.bg_pano_rgb_loss_weight}, sds rgb loss weight {self.sds_rgb_loss_weight} **********')
                self.train_dataset.change_sampling_idx(-1)
                for delta_epoch in range(self.color_epoch_num):

                    for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                        model_input["intrinsics"] = model_input["intrinsics"].cuda()
                        model_input["uv"] = model_input["uv"].cuda()
                        model_input['pose'] = model_input['pose'].cuda()

                        self.optimizer.zero_grad()

                        # random choose anchor views
                        anchor_batch_size = 5
                        merge_model_input = {}
                        merge_gt_rgb = torch.zeros((anchor_batch_size+1, self.img_res[0], self.img_res[1], 3)).cuda()
                        merge_gt_rgb[0] = ground_truth['rgb'].cuda().reshape(self.img_res[0], self.img_res[1], 3)

                        merge_model_input['intrinsics'] = model_input['intrinsics']
                        merge_model_input['pose'] = torch.zeros((anchor_batch_size+1, 4, 4)).cuda()
                        merge_model_input['pose'][0] = model_input['pose'][0]
                        random_anchor_list = np.random.choice(total_anchor_num, anchor_batch_size, replace=False)
                        merge_idx = 1
                        for idx in random_anchor_list:

                            merge_model_input['pose'][merge_idx] = anchor_camera_pose[idx].cuda()

                            nerf_anchor_rgb_path = os.path.join(save_nerf_anchor_root_path, f'nerf_rgb_{idx}.png')
                            nerf_anchor_rgb = Image.open(nerf_anchor_rgb_path)
                            nerf_anchor_rgb = torch.from_numpy(np.array(nerf_anchor_rgb)).float().cuda() / 255.0
                            merge_gt_rgb[merge_idx] = nerf_anchor_rgb

                            merge_idx += 1

                        out = self.model.module.get_scene_rgb(merge_model_input, indices, height=self.img_res[0], width=self.img_res[1], return_all=True)
                        pred_scene_rgb = out['comp_rgb']
                        vis_mask = out["vis_map"] >= self.model.module.bg_vis_thresh
                        vis_mask[0] = torch.ones_like(vis_mask[0])          # input scene image should be all visible

                        pred_scene_rgb = pred_scene_rgb * vis_mask
                        gt_scene_rgb = merge_gt_rgb * vis_mask

                        # # visulize pred and gt rgb
                        # if delta_epoch % 10 == 0:
                        #     for idx in range(anchor_batch_size+1):
                        #         pred_rgb = pred_scene_rgb[idx].detach().cpu().numpy()
                        #         gt_rgb = gt_scene_rgb[idx].detach().cpu().numpy()
                        #         pred_rgb = Image.fromarray((pred_rgb*255).astype('uint8'))
                        #         gt_rgb = Image.fromarray((gt_rgb*255).astype('uint8'))
                        #         temp_visual_path = os.path.join(self.plots_dir, 'nerf_color_anchor_visual')
                        #         os.makedirs(temp_visual_path, exist_ok=True)

                        #         cat_rgb_save_path = os.path.join(temp_visual_path, f'color_opt_{epoch+delta_epoch}_{indices.item()}_{idx}.png')
                        #         cat_rgb = Image.new('RGB', (2*self.img_res[0], self.img_res[1]))
                        #         cat_rgb.paste(pred_rgb, (0, 0))
                        #         cat_rgb.paste(gt_rgb, (self.img_res[0], 0))
                        #         cat_rgb.save(cat_rgb_save_path)

                        scene_rgb_loss = F.mse_loss(pred_scene_rgb, gt_scene_rgb)

                        bg_pano_loss = self.model.module.get_bg_pano_loss()

                        sds_loss_dict = self.model.module.get_color_prior_loss()
                        sd_loss = sds_loss_dict['loss_rgb_sd']

                        loss = scene_rgb_loss * self.scene_rgb_loss_weight + sd_loss * self.sds_rgb_loss_weight + bg_pano_loss * self.bg_pano_rgb_loss_weight             # scene rgb loss should be larger than sd loss
                        loss.backward()

                        self.optimizer.step()
                        self.scheduler.step()

                    print(f'{epoch+delta_epoch} epoch, {indices.item()}, scene_rgb_loss: {scene_rgb_loss.item()}, sd_loss: {sd_loss.item()}')

                # update epoch
                delta_epoch += 1
                epoch += delta_epoch

                # export total scene color mesh
                print('********** color optimization over, export total scene color mesh **********')
                self.model.module.infer_color_anchor_views()                                # infer color anchor views

                if self.GPU_INDEX == 0:
                    self.save_checkpoints(epoch)      # save checkpoints after color optimization

                print('********** begin render nvdiffrast color image **********')
                # infer nerf rgb
                from PIL import Image

                self.model.eval()
                self.visgrid.eval()

                infer_camera_files = os.path.join(self.train_dataset.instance_dir, 'eval-novel-views', 'cameras.npz')               # infer all 100 views
                infer_camera_dict = np.load(infer_camera_files)

                eval_json_path = os.path.join(self.train_dataset.instance_dir, 'eval-novel-views', 'eval-10views.json')             # 10 views for evaluation
                with open(eval_json_path, 'r') as f:
                    eval_view_list = json.load(f)

                infer_views_num = 100
                scale_mats = [infer_camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(infer_views_num)]
                world_mats = [infer_camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(infer_views_num)]
                cam_intrinsics = self.train_dataset.intrinsics_all[0]       # use the same intrinsics for all views
                
                nvdiffrast_rgb_root_path = os.path.join(self.plots_dir, 'eval_nvdiffrast_rgb')
                os.makedirs(nvdiffrast_rgb_root_path, exist_ok=True)

                idx = 0
                for scale_mat, world_mat in zip(scale_mats, world_mats):

                    if idx not in eval_view_list:
                        idx += 1
                        continue

                    model_input = {}

                    P = world_mat @ scale_mat
                    P = P[:3, :4]
                    intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

                    if self.train_dataset.need_fix_y:
                        pose = rot_cameras_along_y(pose, self.train_dataset.fix_y_deg)              # need to fix the camera pose along y axis for replica scan3
                    if self.train_dataset.need_fix_x:
                        pose = rot_cameras_along_x(pose, self.train_dataset.fix_x_deg)              # need to fix the camera pose along x axis for youtube scan1 and scan2

                    pose = torch.from_numpy(pose).float()

                    model_input['pose'] = pose.unsqueeze(0).cuda()
                    model_input['intrinsics'] = cam_intrinsics.unsqueeze(0).cuda().float()

                    # save nvdiffrast rgb
                    mesh_rgb = self.model.module.get_scene_rgb(model_input, indices, height=self.img_res[0], width=self.img_res[1]).detach().cpu().numpy()
                    mesh_rgb = (mesh_rgb * 255).astype(np.uint8)
                    mesh_rgb = Image.fromarray(mesh_rgb)
                    mesh_rgb_path = os.path.join(nvdiffrast_rgb_root_path, f'nvdiffrast_rgb_{epoch}_{idx}.png')
                    mesh_rgb.save(mesh_rgb_path)

                    idx += 1

                print('********** export color mesh **********')
                prior_bbox_root_path = os.path.join(self.plots_dir, 'prior_bbox')
                color_mesh_save_root_path = os.path.join(self.plots_dir, 'color_mesh')
                for obj_idx in self.model.module.prior_obj_idx_list:
                    if obj_idx == 0:
                        prior_bbox_path = None
                    else:
                        prior_bbox_path = os.path.join(prior_bbox_root_path, f'bbox_{obj_idx}.json')
                    self.model.module.mesh_color_network.set_obj_idx_list_bbox(obj_idx)
                    self.model.module.prior.export_color_mesh(obj_idx=obj_idx, save_root_path=color_mesh_save_root_path, save_name=f'color_mesh_e{epoch}_{obj_idx}', prior_bbox_path=prior_bbox_path)
                obj_idx = 1000
                self.model.module.mesh_color_network.set_obj_idx_list_bbox(0)         # set total scene as the background
                self.model.module.prior.export_color_mesh(obj_idx=obj_idx, save_root_path=color_mesh_save_root_path, save_name=f'color_mesh_e{epoch}_{obj_idx}')

                print('********** finish color optimization **********')

                if self.use_wandb:
                    wandb.finish()
                print('training over')
                exit()

            else:                                                               # recon stage and add geometry prior stage

                self.train_dataset.change_sampling_idx(self.num_pixels)

                for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()
                    model_input['pose'] = model_input['pose'].cuda()
                    
                    self.optimizer.zero_grad()
                    
                    model_outputs = self.model(model_input, indices, iter_step=self.iter_step, visgrid=self.visgrid)
                    model_outputs['iter_step'] = self.iter_step
                    
                    loss_output = self.loss(model_outputs, ground_truth, call_reg=True) if\
                            self.iter_step >= self.add_objectvio_iter else self.loss(model_outputs, ground_truth, call_reg=False)
                    # if change the pixel sampling pattern to patch, then you can add a TV loss to enforce some smoothness constraint
                    loss = loss_output['loss']
                    loss.backward()

                    # calculate gradient norm
                    total_norm = 0
                    parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                    for p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5

                    self.optimizer.step()
                    
                    psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                            ground_truth['rgb'].cuda().reshape(-1,3))
                    
                    self.iter_step += 1                
                    
                    if self.GPU_INDEX == 0 and data_index %20 == 0:
                        print(
                            '{0}_{1} [{2}] ({3}/{4}): loss = {5}, rgb_loss = {6}, eikonal_loss = {7}, psnr = {8}, bete={9}, alpha={10}, semantic_loss = {11}, reg_loss = {12}, sds_loss = {13}'
                                .format(self.expname, self.timestamp, epoch, data_index, self.n_batches, loss.item(),
                                        loss_output['rgb_loss'].item(),
                                        loss_output['eikonal_loss'].item(),
                                        psnr.item(),
                                        self.model.module.density.get_beta().item(),
                                        1. / self.model.module.density.get_beta().item(),
                                        loss_output['semantic_loss'].item(),
                                        loss_output['collision_reg_loss'].item(),
                                        loss_output['sds_loss'].item() if 'sds_loss' in loss_output else 0.0))
                        
                        if self.use_tensorboard:
                            for k, v in loss_output.items():
                                self.writer.add_scalar(f'Loss/{k}', v.item(), self.iter_step)
                            self.writer.add_scalar('Statistics/beta', self.model.module.density.get_beta().item(), self.iter_step)
                            self.writer.add_scalar('Statistics/alpha', 1. / self.model.module.density.get_beta().item(), self.iter_step)
                            self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                            self.writer.add_scalar('Statistics/total_norm', total_norm, self.iter_step)
                            
                            if self.Grid_MLP:
                                self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                                self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                                self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)

                        if self.use_wandb:
                            for k, v in loss_output.items():
                                wandb.log({f'Loss/{k}': v.item()}, self.iter_step)

                            wandb.log({'Statistics/beta': self.model.module.density.get_beta().item()}, self.iter_step)
                            wandb.log({'Statistics/alpha': 1. / self.model.module.density.get_beta().item()}, self.iter_step)
                            wandb.log({'Statistics/psnr': psnr.item()}, self.iter_step)
                            wandb.log({'Statistics/total_norm': total_norm}, self.iter_step)
                            
                            if self.Grid_MLP:
                                wandb.log({'Statistics/lr0': self.optimizer.param_groups[0]['lr']}, self.iter_step)
                                wandb.log({'Statistics/lr1': self.optimizer.param_groups[1]['lr']}, self.iter_step)
                                wandb.log({'Statistics/lr2': self.optimizer.param_groups[2]['lr']}, self.iter_step)
                    
                    self.train_dataset.change_sampling_idx(self.num_pixels)
                    self.scheduler.step()

        if self.GPU_INDEX == 0:
            self.save_checkpoints(epoch)

        # if self.GPU_INDEX == 0:
        #     self.model.eval()
        #     surface_traces = plt.get_surface_sliding(path=self.plots_dir,
        #                         epoch=epoch + 1,
        #                         sdf=lambda x: self.model.module.implicit_network.get_sdf_vals(x).reshape(-1),
        #                         resolution=1024,
        #                         grid_boundary=self.plot_conf['grid_boundary'],
        #                         level=0
        #                         )
        
        if self.use_wandb:
            wandb.finish()
        print('training over')

        
    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, seg_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        seg_map = model_outputs['semantic_values'].reshape(batch_size, num_samples)
        seg_gt = seg_gt.to(seg_map.device)

        vis_map = model_outputs['vis_values'].reshape(batch_size, num_samples)

        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'seg_gt': seg_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            'seg_map': seg_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
            'vis_map': vis_map,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()
