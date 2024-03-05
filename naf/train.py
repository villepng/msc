import functools
import math
import numpy as np
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import os

from contextlib import closing
from torch.nn.parallel import DistributedDataParallel as DDP

from data_loading.sound_loader import Soundsamples
from model.modules import EmbeddingModuleLog
from model.networks import KernelResidualFCEmbeds, PhaseLoss
from options import Options


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def worker_init_fn(worker_id, myrank_info):
    # print(worker_id + myrank_info*100, 'SEED')
    np.random.seed(worker_id + myrank_info * 100)


def train_net(rank, world_size, freeport, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    output_device = rank
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    pi = math.pi
    pixel_count = args.pixel_count

    dataset = Soundsamples(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)
    sound_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size // world_size, shuffle=False,
                                               num_workers=3, worker_init_fn=ranked_worker_init,
                                               persistent_workers=True, sampler=train_sampler, drop_last=False)

    xyz_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2, max_freq=7).to(output_device)
    time_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2).to(output_device)
    freq_embedder = EmbeddingModuleLog(num_freqs=args.num_freqs, ch_dim=2).to(output_device)

    auditory_net = KernelResidualFCEmbeds(input_ch=105, intermediate_ch=args.features, grid_ch=args.grid_features, num_block=args.layers,
                                          grid_gap=args.grid_gap, grid_bandwidth=args.bandwith_init, bandwidth_min=args.min_bandwidth,
                                          bandwidth_max=args.max_bandwidth, float_amt=args.position_float, min_xy=dataset.min_pos,
                                          max_xy=dataset.max_pos, components=args.components).to(output_device)

    if rank == 0:
        print(f'Dataloader requires {len(sound_loader)} batches')

    start_epoch = 1
    load_opt = 0
    loaded_weights = False
    if args.resume:
        if not os.path.isdir(args.exp_dir):
            print('Missing save dir, exiting')
            dist.barrier()
            dist.destroy_process_group()
            return 1
        else:
            current_files = sorted(os.listdir(args.exp_dir))
            if len(current_files) > 0:
                latest = current_files[-1]
                start_epoch = int(latest.split('.')[0]) + 1
                if rank == 0:
                    print(f'Identified checkpoint \'{latest}\'')
                if start_epoch >= (args.epochs + 1):
                    dist.barrier()
                    dist.destroy_process_group()
                    return 1
                map_location = 'cuda:%d' % rank
                weight_loc = os.path.join(args.exp_dir, latest)
                weights = torch.load(weight_loc, map_location=map_location)
                if rank == 0:
                    print(f'Checkpoint loaded from \'{weight_loc}\'')
                auditory_net.load_state_dict(weights['network'])
                loaded_weights = True
                if 'opt' in weights:
                    load_opt = 1
                dist.barrier()
        if loaded_weights is False:
            print('Resume indicated, but no weights found')
            dist.barrier()
            dist.destroy_process_group()
            exit()

    # We have conditional forward, must set find_unused_parameters to true
    ddp_auditory_net = DDP(auditory_net, find_unused_parameters=True, device_ids=[rank])
    criterion = torch.nn.MSELoss()
    criterion_phase = PhaseLoss()  # todo
    orig_container = []
    grid_container = []
    for par_name, par_val in ddp_auditory_net.named_parameters():
        if 'grid' in par_name:
            grid_container.append(par_val)
        else:
            orig_container.append(par_val)

    optimizer = torch.optim.AdamW([
        {'params': grid_container, 'lr': args.lr_init, 'weight_decay': 1e-2},
        {'params': orig_container, 'lr': args.lr_init, 'weight_decay': 0.0}], lr=args.lr_init,
        weight_decay=0.0)

    if load_opt:
        print('Loading optimizer')
        optimizer.load_state_dict(weights['opt'])
        dist.barrier()

    progress = tqdm.tqdm(range(start_epoch, args.epochs + 1))
    progress.set_description(f'Starting training for room \'{args.exp_name}\'')
    for epoch in progress:
        total_losses = 0
        cur_iter = 0
        for data_stuff in sound_loader:
            gt = data_stuff[0].to(output_device, non_blocking=True)
            early = data_stuff[1].to(output_device, non_blocking=True)
            degree = data_stuff[2].to(output_device, non_blocking=True)
            position = data_stuff[3].to(output_device, non_blocking=True)
            non_norm_position = data_stuff[4].to(output_device, non_blocking=True)
            freqs = data_stuff[5].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            times = data_stuff[6].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            times_early = data_stuff[7].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi
            # times_ph = data_stuff[7].to(output_device, non_blocking=True).unsqueeze(2) * 2.0 * pi

            with torch.no_grad():
                position_embed = xyz_embedder(position).expand(-1, pixel_count, -1)
                freq_embed = freq_embedder(freqs)
                time_embed = time_embedder(times)
                time_embed_early = time_embedder(times_early)
                # time_embed_ph = time_embedder(times_ph)

            total_in = torch.cat((position_embed, time_embed_early), dim=2)
            optimizer.zero_grad(set_to_none=False)
            out_early = ddp_auditory_net(total_in, non_norm_position.squeeze(1)).squeeze()  # .squeeze(3).transpose(1, 2)
            # output = output.squeeze(3).transpose(1, 2)
            # out_spec = output  # [:, :, :, 0]
            # out_phase = output[:, :, :, 1]
            # a = 200
            # loss = criterion(output, gt)
            loss_early = criterion(out_early, early)
            if rank == 0:
                total_losses += loss_early.detach()
                progress.set_description(f' early loss: {loss_early.detach():.6f}')  # mag loss: {loss.detach():.6f},
                cur_iter += 1
            loss = loss_early
            loss.backward()
            optimizer.step()
        decay_rate = args.lr_decay
        new_lrate_grid = args.lr_init * (decay_rate ** (epoch / args.epochs))
        new_lrate = args.lr_init * (decay_rate ** (epoch / args.epochs))

        par_idx = 0
        for param_group in optimizer.param_groups:
            if par_idx == 0:
                param_group['lr'] = new_lrate_grid
            else:
                param_group['lr'] = new_lrate
            par_idx += 1
        if rank == 0:
            avg_loss = total_losses.item() / cur_iter
            if epoch % 10 == 0 or epoch < 10:
                print(f'\n  Ending epoch {epoch} for room \'{args.exp_name}\', avg. loss {avg_loss:.6f}')
        if rank == 0 and (epoch % 20 == 0 or epoch == 1 or epoch > (args.epochs - 3)):
            save_name = str(epoch).zfill(4) + '.chkpt'
            save_dict = {'network': ddp_auditory_net.module.state_dict()}
            torch.save(save_dict, os.path.join(args.exp_dir, save_name))
    print(f'Wrapping up training for room \'{args.exp_name}\'')
    dist.barrier()
    dist.destroy_process_group()
    return 1


if __name__ == '__main__':
    cur_args = Options().parse()
    exp_name = cur_args.exp_name
    exp_name_filled = exp_name.format(cur_args.apt)
    cur_args.exp_name = exp_name_filled
    if not os.path.isdir(cur_args.model_save_loc):
        print(f'Save directory {cur_args.model_save_loc} does not exist, creating...')
        os.makedirs(cur_args.model_save_loc)
    exp_dir = os.path.join(cur_args.model_save_loc, exp_name_filled)
    cur_args.exp_dir = exp_dir
    print(f'Experiment directory is {exp_dir}')
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    world_size = cur_args.gpus
    myport = str(find_free_port())
    mp.spawn(train_net, args=(world_size, myport, cur_args), nprocs=world_size, join=True)
