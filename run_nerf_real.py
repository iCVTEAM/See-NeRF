import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from run_nerf_helpers import *
from load_llff import pose_interpolation_llff
from load_llff import load_llff_data
from torch.utils.tensorboard import SummaryWriter
import math
from event_loss_helpers import *
from event_loss_helpers import bin_num_eval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, exp_times, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, video=False):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    exp_video = np.linspace(start=exp_times[0], stop=exp_times[4], num=int(len(render_poses) / 2 + 1), endpoint=True)
    reverse = exp_video[::-1]
    exp_video = np.concatenate([exp_video, reverse])
    rgbs_l_video = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, extra = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgb_h = rgb.cpu().numpy()
        disp = disp.cpu().numpy()

        if not video:
            for j in range(5):
                rgb_l = render_kwargs["network_fine"].crf(rgb.reshape([-1, 3]) + math.log(exp_times[j])).reshape([H, W, 3]).cpu().numpy()

                rgb8 = to8b(rgb_l)
                filename = os.path.join(savedir, '{0:03d}_l_{1}.png'.format(i, j))
                imageio.imwrite(filename, rgb8)

                if j==0:
                    rgb8 = to8b(np.exp(rgb_h))
                    filename = os.path.join(savedir, '{:03d}.png'.format(i))
                    imageio.imwrite(filename, rgb8)

                    rgb_h_tm = np.exp(rgb_h)
                    filename = os.path.join(savedir, '{:03d}_h.npy'.format(i))
                    np.save(filename, rgb_h_tm)

                    rgb_h = np.exp(rgb_h).astype(np.float32)
                    filename = os.path.join(savedir, '{:03d}_h.exr'.format(i))
                    rgb_h = rgb_h[:, :, ::-1]
                    cv2.imwrite(filename, rgb_h)

                    rgb8 = to8b(disp / np.max(disp))
                    filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
                    imageio.imwrite(filename, rgb8)


        else:
            rgb_l = render_kwargs["network_fine"].crf(rgb.reshape([-1, 3]) + math.log(exp_times[2])).reshape([H, W, 3]).cpu().numpy()
            rgbs_l_video.append(rgb_l)
            rgb = rgb.cpu().numpy()

            rgb_h_tm = np.exp(rgb)
            filename = os.path.join(savedir, '{:03d}_h.npy'.format(i))
            np.save(filename, rgb_h_tm)

            rgb_h = np.exp(rgb).astype(np.float32)
            filename = os.path.join(savedir, '{:03d}_h.exr'.format(i))
            rgb_h = rgb_h[:, :, : :-1]
            cv2.imwrite(filename, rgb_h)

    if video:
        rgbs_l_video = np.stack(rgbs_l_video, 0)

    return rgbs_l_video


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    #rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    rgb = raw[...,:3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (240/255.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth_map0'] = depth_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: blender / ellff')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # event options
    parser.add_argument("--use_event", type=bool, default=True,
                        help='use event to help the training')
    parser.add_argument("--bin_num_evaluater", action='store_true',
                        help='use event to help the training')
    parser.add_argument("--bin_num", type=int, default=6,
                        help='the number of event bin')
    parser.add_argument("--pre_iters", type=int, default=0,
                        help='number of steps to train without bin_num_evaluater')
    parser.add_argument("--exp_time", type=float, default=0.08,
                        help='the exposure time of the images')
    parser.add_argument("--cutoff_hz", type=int, default=30,
                        help='cutoff_hz of event sensor')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=10,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_val",     type=int, default=100000,
                        help='frequency of render pose[0] image')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def setup_seed(seed):
    #  下面两个常规设置了，用来np和random的话要设置
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。


def train():
    print("")
    print("-----------------starting------------------")
    parser = config_parser()
    args = parser.parse_args()
    setup_seed(0)

    print("---------------data_loading----------------")
    K = None

    if args.dataset_type == 'ellff':
        res_height = 260
        res_width = 346
        res = res_height * res_width
        blender = False

        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]

        new_poses = []
        new_images = []
        test_poses = []
        test_poses_new = poses[images.shape[0] * (args.bin_num + 1):]
        for i in range(images.shape[0]):
            new_images.append(images[i])
            pose = []
            test_poses.append(poses[i * (args.bin_num + 1)])
            for j in range(args.bin_num + 1):
                pose.append(poses[i * (args.bin_num + 1) + j])
            new_poses.append(np.stack(pose, axis=0))
        poses = np.stack(new_poses, axis=0)
        images = np.stack(new_images, axis=0)
        test_poses = np.stack(test_poses, axis=0)

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if args.bin_num_evaluater:
            poses = [poses[:,[0,4,8,12,16],:,:], poses[:,[0,2,4,6,8,10,12,14,16],:,:],
                     poses[:,[0,1,3,4,5,7,8,9,11,12,13,15,16]]]

        # poses_extra = poses[:, 0, :, :] + (poses[:, 0, :, :] - poses[:, 1, :, :])
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        exp_times_test = np.loadtxt(os.path.join(args.datadir + "/exp_times_test.npy"))
        exp_times_train = np.ones(images.shape[0]) * exp_times_test[2]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])


    print("------------event_data_loading------------")
    use_event = args.use_event
    if use_event:
        # load event
        event_maps = torch.load(os.path.join(args.datadir, "events_offset.pt"))
        # load frame weights
        frames_weights = np.loadtxt(os.path.join(args.datadir, "frames_weights.npy"))
        # load masks
        event_mask = np.ones((images.shape[0], res_height, res_width))
        # color masks
        color_masks = np.zeros((res_height, res_width, 3))
        color_masks[0::2, 0::2, 0] = 1  # r
        color_masks[0::2, 1::2, 1] = 1  # g
        color_masks[1::2, 0::2, 1] = 1  # g
        color_masks[1::2, 1::2, 2] = 1  # b
        color_masks = torch.tensor(color_masks.reshape((-1, 3))).to(device).float()

    print("--------------creat_nerf--------------")
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    # CRF initialization
    if start == 0:
        render_kwargs_train['network_fn'] = warm_crf(render_kwargs_train['network_fn'], pretrain_iters=500)
        render_kwargs_train['network_fn'] = warm_crf_evs(render_kwargs_train['network_fn'], pretrain_iters=500)
        render_kwargs_train['network_fine'] = warm_crf(render_kwargs_train['network_fine'], pretrain_iters=500)
        render_kwargs_train['network_fine'] = warm_crf_evs(render_kwargs_train['network_fine'], pretrain_iters=500)

    draw_CRF(os.path.join(basedir, expname), render_kwargs_test['network_fine'])
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move training data to GPU
    N_rand = args.N_rand
    images = torch.Tensor(images).to(device)

    # bin_num_evaluater
    if args.bin_num_evaluater:
        bin_num_evaluater = bin_num_eval(event_map_4.shape[0], os.path.join(basedir, expname))

    print("---------------start_training----------------")

    N_iters = 50000 + 1
    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, expname, 'logs'))

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        img_i = random.randrange(images.shape[0])
        target = images[img_i]
        exp = exp_times_train[img_i]

        if args.bin_num_evaluater:
            if i >= args.pre_iters:
                if blender:
                    bin_flag = bin_num_evaluater.get_bin_flag(img_i)
                else:
                    bin_flag = bin_num_evaluater.get_bin_flag_ellff(img_i)
            else:
                bin_flag = 0
            event_map = event_maps[bin_flag][img_i]
            frames_weight = frames_weights[bin_flag][img_i]
            pose = poses[bin_flag][img_i, :, :3, :4]
            bin_num = [4,6,8][bin_flag]
        else:
            event_map = event_maps[img_i]
            # flow_map = flow_maps[img_i]
            frames_weight = frames_weights[img_i]
            pose = poses[img_i, :, :3, :4]
            bin_num = args.bin_num

        rays_os = []
        rays_ds = []
        for j in range(bin_num + 1):
            ray_o, ray_d = get_rays(H, W, K, torch.Tensor(pose[j]))
            rays_os.append(ray_o)
            rays_ds.append(ray_d)

        if N_rand is not None:
            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),-1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2]).long()   # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)

        mask_sharp_inds = np.nonzero(np.logical_not(event_mask[img_i]).reshape(res))
        mask_blur_inds = np.nonzero(event_mask[img_i].reshape(res))

        blur_inds = np.intersect1d(select_inds, mask_blur_inds, assume_unique=True)
        sharp_inds = np.intersect1d(select_inds, mask_sharp_inds, assume_unique=True)

        # blur rendering
        if len(blur_inds) != 0:
            select_blur_coords = coords[blur_inds]
            color_mask = color_masks[blur_inds]

            rays_o = torch.stack(rays_os)
            rays_d = torch.stack(rays_ds)
            rays_o = rays_o[:, select_blur_coords[:, 0], select_blur_coords[:, 1]].view((-1, 3)) # (N_rand, 3)
            rays_d = rays_d[:, select_blur_coords[:, 0], select_blur_coords[:, 1]].view((-1, 3)) # (N_rand, 3)

            batch_rays = torch.stack([rays_o, rays_d], 0)
            rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                        verbose=i < 10, retraw=True,
                                                        **render_kwargs_train)


            blur_rgbs = [rgb.view((bin_num + 1), -1, 3)[j] for j in range(bin_num + 1)]
            blur_rgb = [torch.exp(rgb).view((bin_num + 1), -1, 3)[j] * frames_weight[j] for j in range(bin_num + 1)]
            blur_rgbs_extras = [extras['rgb0'].view((bin_num + 1), -1, 3)[j] for j in range(bin_num + 1)]
            blur_rgb_extras = [torch.exp(extras['rgb0']).view((bin_num + 1), -1, 3)[j] * frames_weight[j] for j in range(bin_num + 1)]
            blur_depthes = [depth.view((bin_num + 1), -1, 1)[j] for j in range(bin_num + 1)]
            blur_depthes_extras = [extras['depth_map0'].view((bin_num + 1), -1, 1)[j] for j in range(bin_num + 1)]

            target_blur = target[select_blur_coords[:, 0], select_blur_coords[:, 1]]
            blur_rgbs = torch.stack(blur_rgbs, dim=0)
            blur_rgb = torch.sum(torch.stack(blur_rgb,dim=0), dim=0)
            blur_rgb = render_kwargs_train["network_fine"].crf(torch.log(blur_rgb) + math.log(exp))

            event_data = event_map.view(bin_num, res_height, res_width)[:, select_blur_coords[:, 0], select_blur_coords[:, 1]].to(device).float()
            event_loss = event_loss_call(blur_rgbs, event_data, color_mask, bin_num, render_kwargs_train["network_fine"], exp / bin_num, cutoff_hz=args.cutoff_hz) * 0.005

            img_loss_blur = img2mse(blur_rgb, target_blur)


            if 'rgb0' in extras:
                blur_rgbs_extras = torch.stack(blur_rgbs_extras, dim=0)
                blur_rgb_extras = torch.sum(torch.stack(blur_rgb_extras, dim=0), dim=0)
                blur_rgb_extras = render_kwargs_train["network_fn"].crf(torch.log(blur_rgb_extras) + math.log(exp))

                event_loss0 = event_loss_call(blur_rgbs_extras, event_data, color_mask, bin_num, render_kwargs_train["network_fn"], exp / bin_num, cutoff_hz=args.cutoff_hz) * 0.005

                img_loss0_blur = img2mse(blur_rgb_extras, target_blur)

            if args.bin_num_evaluater and 10000 <= i < args.pre_iters:
                pose_w2c = []
                for j in range(5):
                    pose_w2c.append(np.linalg.inv(pose[j][ :3, :3]))
                bin_num_evaluater.mean_pixels_offset_cal(pose_w2c, K, blur_depthes, select_blur_coords,
                                                         rays_os, rays_ds, img_i, near, far)

        else:
            event_loss = torch.zeros((1))
            event_loss0 = torch.zeros((1))
            img_loss_blur = torch.zeros((1))
            img_loss0_blur = torch.zeros((1))

        if len(sharp_inds) != 0:
            select_sharp_coords = coords[sharp_inds]
            j = random.sample(list(range(bin_num + 1)), 1)
            rays_o = rays_os[j[0]][select_sharp_coords[:, 0], select_sharp_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_ds[j[0]][select_sharp_coords[:, 0], select_sharp_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

            target_sharp = target[select_sharp_coords[:, 0], select_sharp_coords[:, 1]]
            rgb = render_kwargs_train["network_fine"].crf(rgb + math.log(exp))
            img_loss_sharp = img2mse(rgb, target_sharp)

            if 'rgb0' in extras:
                rgb_extras = render_kwargs_train["network_fn"].crf(extras['rgb0'] + math.log(exp))
                img_loss0_sharp = img2mse(rgb_extras, target_sharp)

        else:
            img_loss_sharp = torch.zeros((1))
            img_loss0_sharp = torch.zeros((1))

        unit_exp_loss = point_constraint(render_kwargs_train['network_fine'], 0.5) * 0.5
        unit_exp_loss0 = point_constraint(render_kwargs_train["network_fn"], 0.5) * 0.5

        loss = event_loss + event_loss0 + unit_exp_loss + unit_exp_loss0 +\
               (img_loss_blur * len(blur_inds) + img_loss_sharp * len(sharp_inds)) / N_rand + \
               (img_loss0_blur * len(blur_inds) + img_loss0_sharp * len(sharp_inds)) / N_rand
        #loss = event_loss + img_loss_blur + img_loss_sharp + img_loss0_blur + img_loss0_sharp

        psnr = mse2psnr((img_loss_blur * len(blur_inds) + img_loss_sharp * len(sharp_inds)) / N_rand)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time()-time0



        # Rest is logging
        if i%args.i_weights==0:
            if args.bin_num_evaluater:
                bin_num_evaluater.save()
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            render_poses = pose_interpolation_llff(test_poses_new, 5)
            videosavedir = os.path.join(basedir, expname, 'video_{:06d}'.format(i))
            os.makedirs(videosavedir, exist_ok=True)
            with torch.no_grad():
                rgbs_l_video = render_path(torch.tensor(render_poses), hwf, K, exp_times_test, args.chunk, render_kwargs_test, savedir=videosavedir, video=True)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb_l_video.mp4', to8b(rgbs_l_video), fps=30, quality=8)

        if i%args.i_val==0 and i > 0:
            index = int(i / 1000 - 1)
            val_poses = [render_poses[index%render_poses.shape[0]]]
            valsavedir = os.path.join(basedir, expname, 'valset')
            os.makedirs(valsavedir, exist_ok=True)
            with torch.no_grad():
                rgbs, disps, rgbs_l, _ = render_path(torch.Tensor(val_poses).to(device), hwf, K, exp_times_test, args.chunk, render_kwargs_test)
                # psnr = mse2psnr(img2mse(rgbs[0], gt[0]))
            filename = os.path.join(valsavedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, to8b(rgbs[0]))
            filename = os.path.join(valsavedir, '{:03d}_l.png'.format(i))
            imageio.imwrite(filename, to8b(rgbs_l[0]))
            print('Saved val image_{:06d}'.format(i))

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            os.makedirs(testsavedir + "_novel", exist_ok=True)
            draw_CRF(testsavedir, render_kwargs_test['network_fine'])
            print('test poses shape', test_poses.shape)
            with torch.no_grad():
                render_path(torch.Tensor(np.array(test_poses)).to(device), hwf, K, exp_times_test, args.chunk, render_kwargs_test, savedir=testsavedir)
                render_path(torch.Tensor(np.array(test_poses_new)).to(device), hwf, K, exp_times_test, args.chunk, render_kwargs_test, savedir=testsavedir + "_novel")
            print('Saved test set')

        if i%args.i_print==0:
            writer.add_scalar("loss", loss, i)
            writer.add_scalar("event_loss", event_loss, i)
            writer.add_scalar("event_loss0", event_loss0, i)
            writer.add_scalar("img_loss_blur", img_loss_blur, i)
            writer.add_scalar("img_loss0_blur", img_loss0_blur, i)
            writer.add_scalar("img_loss_sharp", img_loss_sharp, i)
            writer.add_scalar("img_loss0_sharp", img_loss0_sharp, i)
            writer.add_scalar("unit_exp_loss", unit_exp_loss, i)
            writer.add_scalar("unit_exp_loss0", unit_exp_loss0, i)
            #writer.add_scalar("depth_loss", depth_loss, i)
            #writer.add_scalar("depth_loss0", depth_loss0, i)

            writer.add_scalar("psnr", psnr, i)
            #tqdm.write(f"[TRAIN] Iter: {i} Depth_loss: {depth_loss.item()}  Depth_loss0: {depth_loss0.item()}")
            tqdm.write(f"[TRAIN] Iter: {i} Event_Loss: {event_loss.item()}  img_loss_blur: {img_loss_blur.item()} img_loss_sharp:  {img_loss_sharp.item()} ")
            tqdm.write(f"[TRAIN] Iter: {i} Event_Loss0: {event_loss0.item()} img_loss0_blur: {img_loss0_blur.item()}  img_loss0_sharp: {img_loss0_sharp.item()} ")
            tqdm.write(f"[TRAIN] Iter: {i} Bin_num: {bin_num} Img_i: {img_i} Loss: {loss.item()} PSNR: {psnr.item()}")

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
