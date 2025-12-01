
"""Generate lerp videos using pretrained network pickle."""

import os
os.environ["PYTORCH_ROCM_F64"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_DTYPE_DEFAULT"] = "float32"
os.environ["HIPBLAS_TENSILE_LIBPATH"] = ""
os.environ["ROCBLAS_LAYER"] = "NONE"
os.environ["HIP_LAUNCH_BLOCKING"] = "1"


import torch
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_flush_denormal(True)

import builtins
_real_torch_tensor = torch.tensor
def _float_tensor_patch(data, *args, **kwargs):
    kwargs["dtype"] = torch.float32
    return _real_torch_tensor(data, *args, **kwargs)
builtins.torch_tensor = _float_tensor_patch

import re
from typing import List, Optional, Tuple, Union
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
from tqdm import tqdm
import mrcfile
import legacy
from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator


# ----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    """Safely layout image grid for video generation."""
    assert isinstance(img, torch.Tensor), f"[ERROR] layout_grid expected torch.Tensor, got {type(img)}"
    #print(f"[DEBUG] layout_grid: input dtype={img.dtype}, shape={tuple(img.shape)}")

    # Enforce float32 to avoid ROCm FP64 issues
    img = img.to(dtype=torch.float32)

    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = max(1, batch_size // max(grid_h, 1))

    assert batch_size == grid_w * grid_h, (
        f"[ERROR] layout_grid size mismatch: batch={batch_size}, grid={grid_w}x{grid_h}"
    )

    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)

    if chw_to_hwc:
        img = img.permute(1, 2, 0)

    if to_numpy:
        img = img.cpu().numpy().astype(np.uint8 if float_to_uint8 else np.float32)

    #print(f"[DEBUG] layout_grid: output type={type(img)}, dtype={getattr(img, 'dtype', None)}, shape={np.shape(img)}")
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    """Generate voxel grid samples safely in float32 for ROCm."""
    print(f"[DEBUG] create_samples: N={N}, voxel_origin={voxel_origin}, cube_length={cube_length}")

    voxel_origin = np.array(voxel_origin, dtype=np.float32) - cube_length / 2
    voxel_size = np.float32(cube_length / (N - 1))

    overall_index = torch.arange(0, N ** 3, dtype=torch.int64)
    samples = torch.zeros((N ** 3, 3), dtype=torch.float32)

    # Compute voxel coordinates
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples = samples.unsqueeze(0).contiguous().float()

    print(f"[DEBUG] create_samples: samples dtype={samples.dtype}, shape={samples.shape}")
    print(f"[DEBUG] voxel_origin={voxel_origin}, voxel_size={voxel_size}")
    return samples, voxel_origin, voxel_size

# ----------------------------------------------------------------------------

def gen_interp_video(
    G,
    latent,
    mp4: str,
    w_frames=60 * 4,
    kind='cubic',
    grid_dims=(1, 1),
    num_keyframes=None,
    wraps=2,
    psi=1,
    truncation_cutoff=14,
    cfg='FFHQ',
    image_mode='image',
    gen_shapes=False,
    device=torch.device('cuda'),
    **video_kwargs
):
    print(f"[DEBUG] gen_interp_video() start — dtype={torch.get_default_dtype()}, device={device}")
    print(f"[DEBUG] latent type={type(latent)}, shape={getattr(latent, 'shape', None)}, dtype={getattr(latent, 'dtype', None)}")

    grid_w, grid_h = grid_dims
    name = mp4[:-4]

    if num_keyframes is None:
        num_keyframes = max(1, 1 // max(grid_w * grid_h, 1))
        print(f"[DEBUG] num_keyframes auto-set to {num_keyframes}")

    camera_lookat_point = torch.tensor(
        [0, 0, 0.2] if cfg == 'FFHQ' else [0, 0, 0],
        device=device, dtype=torch.float32
    )

    cam2world_pose = LookAtPoseSampler.sample(
        3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device
    ).float()

    intrinsics = torch.tensor(
        [[4.2647, 0, 0.5],
         [0, 4.2647, 0.5],
         [0, 0, 1]],
        device=device, dtype=torch.float32
    )

    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    c = c.repeat(latent.shape[0], 1).float()
    ws = latent.float()

    print(f"[DEBUG] ws shape before adjust: {ws.shape}, target num_ws: {getattr(G.backbone.mapping, 'num_ws', 'unknown')}")

    if hasattr(G.backbone.mapping, 'num_ws') and ws.shape[1] != G.backbone.mapping.num_ws:
        ws = ws.repeat([1, G.backbone.mapping.num_ws, 1])
        print(f"[DEBUG] ws repeated to shape {ws.shape}")

    _ = G.synthesis(ws[:1].float(), c[:1].float())  # warmup to trigger JIT kernels safely

    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    print(f"[DEBUG] ws reshaped: {ws.shape}")

    # Interpolation setup
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy().astype(np.float32), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    max_batch = 10_000_000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = f'interpolation_{name}/'
        os.makedirs(outdir, exist_ok=True)

    all_poses, camera_rotation = [], []

    print("[DEBUG] Starting video frame generation loop...")
    total_frames = num_keyframes * w_frames
    for frame_idx in tqdm(range(total_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range, yaw_range = 0.25, 0.35

                cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / total_frames),
                    3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / total_frames),
                    camera_lookat_point, radius=2.7, device=device
                ).float()

                camera_rotation.append([
                    yaw_range * np.sin(2 * 3.14 * frame_idx / total_frames),
                    -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / total_frames)
                ])

                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = intrinsics.float()
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1).float()

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device=device, dtype=torch.float32)
                img = G.synthesis(ws=w.unsqueeze(0).float(), c=c[0:1].float(), noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img.float())

                if gen_shapes:
                    #print(f"[DEBUG] Generating shape for frame {frame_idx}/{total_frames}")
                    samples, voxel_origin, voxel_size = create_samples(
                        N=voxel_resolution, voxel_origin=[0, 0, 0],
                        cube_length=G.rendering_kwargs['box_warp']
                    )
                    samples = samples.to(device=device, dtype=torch.float32)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device, dtype=torch.float32)
                    transformed_ray_directions_expanded = torch.zeros(
                        (samples.shape[0], max_batch, 3), device=device, dtype=torch.float32
                    )
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm(total=samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                sigma = G.sample_mixed(
                                    samples[:, head:head + max_batch].float(),
                                    transformed_ray_directions_expanded[:, :samples.shape[1] - head].float(),
                                    w.unsqueeze(0).float(),
                                    truncation_psi=psi,
                                    noise_mode='const'
                                )['sigma'].float()
                                sigmas[:, head:head + max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)
                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = sigmas[-pad:] = sigmas[:, :pad] = sigmas[:, -pad_top:] = sigmas[:, :, :pad] = sigmas[:, :, -pad:] = 0

                    output_ply = False
                    if output_ply:
                        from shape_utils import convert_sdf_samples_to_ply
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                                   os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                    else:
                        with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc',
                                              overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                            mrc.data[:] = sigmas

        video_out.append_data(layout_grid(torch.stack(imgs).float(), grid_w=grid_w, grid_h=grid_h))

    video_out.close()
    all_poses = np.stack(all_poses)
    camera_rotation = np.array(camera_rotation)

    if gen_shapes:
        print(f"[DEBUG] Saving camera trajectory — {all_poses.shape}")
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)

    print("[DEBUG] gen_interp_video() completed successfully")



# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    """Parse a comma-separated list or range of integers safely.

    Example:
        '1,2,5-10' → [1, 2, 5, 6, 7, 8, 9, 10]
    """
    print(f"[DEBUG] parse_range() called with input: {s} ({type(s).__name__})")

    try:
        if isinstance(s, list):
            print("[DEBUG] Input already a list → returning as-is")
            return [int(x) for x in s]

        if not isinstance(s, str):
            raise TypeError(f"Expected str or list[int], got {type(s)}")

        ranges = []
        range_re = re.compile(r'^(\d+)-(\d+)$')
        for p in s.split(','):
            p = p.strip()
            if not p:
                continue
            match = range_re.match(p)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                if start > end:
                    print(f"[WARN] Invalid range {start}-{end}, skipping")
                    continue
                ranges.extend(range(start, end + 1))
            else:
                try:
                    ranges.append(int(p))
                except ValueError:
                    print(f"[WARN] Skipping invalid token '{p}' in input string")
        print(f"[DEBUG] Parsed result: {ranges}")
        return ranges

    except Exception as e:
        print(f"[ERROR] parse_range() failed: {e}")
        return []


def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple safely.

    Example:
        '4x2' → (4, 2)
        '0,1' → (0, 1)
    """
    print(f"[DEBUG] parse_tuple() called with input: {s} ({type(s).__name__})")

    try:
        if isinstance(s, tuple):
            if len(s) == 2 and all(isinstance(x, int) for x in s):
                print("[DEBUG] Input already a valid tuple → returning as-is")
                return s
            raise ValueError(f"Tuple must contain exactly two integers, got {s}")

        if not isinstance(s, str):
            raise TypeError(f"Expected str or tuple[int,int], got {type(s)}")

        match = re.match(r'^(\d+)[x,](\d+)$', s.strip())
        if not match:
            raise ValueError(f"Cannot parse tuple from string: '{s}'")

        result = (int(match.group(1)), int(match.group(2)))
        print(f"[DEBUG] Parsed tuple result: {result}")
        return result

    except Exception as e:
        print(f"[ERROR] parse_tuple() failed: {e}")
        return (0, 0)


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--npy_path', 'npy_path', help='Network pickle filename', required=True)
@click.option('--num-keyframes', type=int,
              help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
              default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR',
              default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def generate_images(
        network_pkl: str,
        npy_path: str,
        truncation_psi: float,
        truncation_cutoff: int,
        num_keyframes: Optional[int],
        w_frames: int,
        outdir: str,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
):
    """Render a latent vector interpolation video (ROCm-safe version)."""

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Loading network from: {network_pkl}")
    device = torch.device('cuda')

    # Disable any chance of FP64 usage globally
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_flush_denormal(True)

    print(f"[DEBUG] Torch version: {torch.__version__}")
    print(f"[DEBUG] Default dtype: {torch.get_default_dtype()}")
    print(f"[DEBUG] Available device: {torch.cuda.get_device_name(0)}")

    # --- Load model ---
    if 'pkl' in network_pkl:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device).float()
        print(f"[DEBUG] Loaded network from pickle → dtype={next(G.parameters()).dtype}")

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    else:
        print("[INFO] Reloading TriPlaneGenerator...")
        init_kwargs = {
            'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2},
            'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only',
            'rendering_kwargs': {
                'depth_resolution': 48, 'depth_resolution_importance': 48,
                'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
                'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
                'disparity_space_sampling': False, 'clamp_mode': 'softplus',
                'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
                'c_gen_conditioning_zero': False, 'c_scale': 1.0,
                'superresolution_noise_mode': 'none', 'density_reg': 0.25,
                'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
                'sr_antialias': True
            },
            'num_fp16_res': 0, 'sr_num_fp16_res': 4,
            'sr_kwargs': {'channel_base': 32768, 'channel_max': 512,
                          'fused_modconv_default': 'inference_only'},
            'conv_clamp': None, 'c_dim': 25, 'img_resolution': 512, 'img_channels': 3
        }

        rendering_kwargs = {
            'depth_resolution': 96, 'depth_resolution_importance': 96,
            'ray_start': 2.25, 'ray_end': 3.3, 'box_warp': 1,
            'avg_camera_radius': 2.7, 'avg_camera_pivot': [0, 0, 0.2],
            'image_resolution': 512, 'disparity_space_sampling': False,
            'clamp_mode': 'softplus', 'superresolution_module': 'training.superresolution.SuperresolutionHybrid8XDC',
            'c_gen_conditioning_zero': False, 'c_scale': 1.0,
            'superresolution_noise_mode': 'none', 'density_reg': 0.25,
            'density_reg_p_dist': 0.004, 'reg_type': 'l1',
            'decoder_lr_mul': 1.0, 'sr_antialias': True
        }

        G = TriPlaneGenerator(**init_kwargs).eval().requires_grad_(False).to(device).float()
        ckpt = torch.load(network_pkl, map_location='cpu')
        G.load_state_dict(ckpt['G_ema'], strict=False)
        G = G.float()
        G.neural_rendering_resolution = 128
        G.rendering_kwargs = rendering_kwargs

        # Debugging
        print(f"[DEBUG] Model loaded manually → dtype={next(G.parameters()).dtype}")

        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance'] = int(
            G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

    # --- Post-load adjustments ---
    if nrr is not None:
        print(f"[DEBUG] Overriding neural rendering resolution to {nrr}")
        G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0
    if truncation_psi == 1.0:
        truncation_cutoff = 14

    # --- Latent loading ---
    latent = np.load(npy_path)
    latent = torch.tensor(latent, dtype=torch.float32, device=device)
    print(f"[DEBUG] Latent loaded: shape={latent.shape}, dtype={latent.dtype}, device={latent.device}")

    # --- Sanity check for FP64 tensors ---
    for name, param in G.named_parameters():
        if param.dtype == torch.float64:
            print(f"[WARN] {name} is FP64 → converting to FP32")
            param.data = param.data.float()

    name = os.path.basename(npy_path)[:-4]
    output = os.path.join(outdir, f'{name}.mp4')
    print(f"[INFO] Generating video: {output}")

    try:
        gen_interp_video(
            G=G,
            latent=latent,
            mp4=output,
            bitrate='10M',
            grid_dims=(1, 1),
            num_keyframes=num_keyframes,
            w_frames=w_frames,
            psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            cfg=cfg,
            image_mode=image_mode
        )
    except RuntimeError as e:
        print(f"[ERROR] Runtime failure → {e}")
        print("[DEBUG] Attempting to dump dtype info:")
        for n, p in G.named_parameters():
            print(f"   {n}: {p.dtype}")
        raise



# ----------------------------------------------------------------------------

if __name__ == "__main__":

    print("Torch version:", torch.__version__)
    print("Default dtype:", torch.get_default_dtype())
    print("Device dtype:", torch.randn(1, device='cuda').dtype)
    print("FP64 allowed:", os.environ.get("PYTORCH_ROCM_F64"))

    generate_images()

# ----------------------------------------------------------------------------