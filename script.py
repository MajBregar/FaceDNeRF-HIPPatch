import os

import warnings

warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import torch

# Detect available GPUs
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA-compatible GPU detected.")
num_gpus = torch.cuda.device_count()
gpu_info = [(i, torch.cuda.get_device_name(i)) for i in range(num_gpus)]

print("Detected GPUs:")
for i, name in gpu_info:
    print(f"  GPU {i}: {name}")

gpu_ids_str = ",".join(str(i) for i, _ in gpu_info)

# Parameters
sd_fine_tune_iterations = 1
sd_generated_poses = 1
dif_mult = 1 #higher numbers require more vram
video_mult = 3
output_dir = "./output/"

image_ids = ["001", "002"]
input_dict = {
        "text": "A human being", 
        "lamda_id": 1.0, #0.2, 
        "lamda_origin": 1.0, #0.2, 
        "lamda_diffusion": 0.0, #1.3e-5, 
        "lamda_illumination": 0.0
}


network_path = "./networks/ffhqrebalanced512-128.pkl"

for image_id in image_ids:

    print("Start:", time.ctime())
    print("Output directory:", output_dir)

    command = (
        f"PYTHONWARNINGS=\"ignore\" "
        f"CUDA_VISIBLE_DEVICES={gpu_ids_str} python run.py "
        f"--outdir='{output_dir}' "
        f"--network='{network_path}' "
        f"--sample_mult={dif_mult} "
        f"--image_path ./test_data/{image_id}.png "
        f"--c_path ./test_data/{image_id}.npy "
        f"--num_steps {sd_fine_tune_iterations} "
        f"--num_steps_pti {sd_generated_poses} "
        f"--description '{input_dict['text']}' "
        f"--lamda_id {input_dict['lamda_id']} "
        f"--lamda_origin {input_dict['lamda_origin']} "
        f"--lamda_diffusion {input_dict['lamda_diffusion']} "
        f"--lamda_illumination {input_dict['lamda_illumination']}"
    )
    print(command)
    os.system(command)

    print("Phase 1 End:", time.ctime())
    print("")
    print("")



    output_dir_video = os.path.join(
        output_dir,
        f"{image_id}_{input_dict['text'].replace(' ', '_')}_{input_dict['lamda_id']}_{input_dict['lamda_origin']}_{input_dict['lamda_diffusion']}_{input_dict['lamda_illumination']}",
    )

    video_command = (
        f"PYTHONWARNINGS=\"ignore\" "
        f"python gen_videos_from_given_latent_code.py "
        f"--outdir='{output_dir_video}' "
        f"--trunc=0.7 "
        f"--npy_path '{output_dir_video}/checkpoints/{image_id}.npy' "
        f"--network='{output_dir_video}/checkpoints/fintuned_generator.pkl' "
        f"--sample_mult={video_mult}"
    )

    print(video_command)
    os.system(video_command)
