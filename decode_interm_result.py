"""
    This script is a skeleton file for **Taihui** to:

    1) Read in the watermark evasion interm. results

    2) Decode each of the interm. result using the encoder/decoder API

    3) Save the result with standardized format
"""
import sys, os
dir_path = os.path.abspath(".")
sys.path.append(dir_path)
dir_path = os.path.abspath("..")
sys.path.append(dir_path)

import argparse, pickle, os, cv2, torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from torchvision import transforms
from PIL import Image
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

# =====
from general import watermark_str_to_numpy, watermark_np_to_str, uint8_to_float, compute_ssim
from optim_utils import transform_img, get_watermarking_pattern, get_watermarking_mask


def calc_mse(img_1_bgr_uint8, img_2_bgr_uint8):
    img_1_float = uint8_to_float(img_1_bgr_uint8)
    img_2_float = uint8_to_float(img_2_bgr_uint8)
    mse = np.mean((img_1_float - img_2_float)**2)
    return mse


def main(args):
    # === This is where the interm. results are saved ===
    # data_root_dir = os.path.join(
    #     "..", "DIP_Watermark_Evasion", "Result-Interm", 
    #     args.watermarker, args.dataset, args.evade_method, args.arch
    # )
    data_root_dir = os.path.join(
        "..", "DIP_Watermark_Evasion", "Result-Interm", 
        args.watermarker, args.dataset, args.evade_method, args.arch
    )
    file_names = [f for f in os.listdir(data_root_dir) if ".pkl" in f]  # Data are saved as dictionary in pkl format.

    # === This is where the watermarked image is stored ===
    im_w_root_dir = os.path.join("dataset", args.watermarker, args.dataset, "encoder_img")

    # === Save the result in a different location in case something went wrong ===
    save_root_dir = os.path.join(
        "..", "DIP_Watermark_Evasion",
        "Result-Decoded", args.watermarker, args.dataset, args.evade_method, args.arch)
    os.makedirs(save_root_dir, exist_ok=True)
    
    # === Process each file ===
    for img_idx, file_name in enumerate(file_names):
        if img_idx <= args.start:
            print("Skip [{}]-th file {}".format(img_idx, file_name))
        elif img_idx > args.end:
            return
        else:
            # Retrieve the im_w name
            im_w_file_name = file_name.replace(".pkl", ".png")
            im_orig_name = im_w_file_name

            # Readin the intermediate files
            data_file_path = os.path.join(data_root_dir, file_name)
            with open(data_file_path, 'rb') as handle:
                data_dict = pickle.load(handle)
            # Readin the im_w into bgr uint8 format
            im_w_path = os.path.join(im_w_root_dir, im_w_file_name)
            im_w_bgr_uint8 = cv2.imread(im_w_path)
            
            # Get the reconstructed data from the interm. result
            if args.evade_method == "WevadeBQ":
                img_recon_list = data_dict["best_recon"]
            else:
                img_recon_list = data_dict["interm_recon"]  # A list of recon. image in "bgr uint8 np" format (cv2 standard format)
            n_recon = len(img_recon_list)
            print("Total number of interm. recon. to process: [{}]".format(n_recon))

            # === Initiate a encoder & decoder ===
            watermark_gt_str = data_dict["watermark_gt_str"]

            # === Init Watermarker (Tree-Ring) ===
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                args.model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
                )
            pipe = pipe.to(device)
            tester_prompt = '' # assume at the detection time, the original prompt is unknown
            text_embeddings = pipe.get_text_embedding(tester_prompt)
            # ground-truth patch
            gt_patch = get_watermarking_pattern(pipe, args, device)
            init_latents_w = pipe.get_random_latents()
            watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

            # Process each inter. recon
            watermark_decoded_log = []  # A list to save decoded watermark
            index_log = data_dict["index"]
            psnr_orig_log = []
            mse_orig_log = []
            psnr_w_log = []
            mse_w_log = []
            ssim_orig_log = []
            ssim_w_log = []
            for img_idx in range(n_recon):
                img_bgr_uint8 = img_recon_list[img_idx]    # shape [512, 512, 3]
                if args.watermarker == "StegaStamp" and args.arch in ["cheng2020-anchor", "mbt2018"]:
                    img_bgr_uint8 = cv2.resize(img_bgr_uint8, (400, 400), interpolation=cv2.INTER_LINEAR)

                # =================== YOUR CODE HERE =========================== #
                
                # Step 0: if you need to change the input format
                img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
                img_input = Image.fromarray(img_rgb_uint8)
                img_input = transform_img(img_input).unsqueeze(0).to(text_embeddings.dtype).to(device)
                image_latents_w = pipe.get_image_latents(img_input, sample=False)

                # Step 1: Decode the interm. result
                img_input_no_w = Image.fromarray(img_rgb_uint8)
                img_no_w_input = transform_img(img_input_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
                image_latents_no_w = pipe.get_image_latents(img_no_w_input, sample=False)
                reversed_latents_w = pipe.forward_diffusion(
                    latents=image_latents_w,
                    text_embeddings=text_embeddings,
                    guidance_scale=1,
                    num_inference_steps=args.test_num_inference_steps,
                )

                # Step 2: log the result
                reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
                target_patch = gt_patch
                w_metric = torch.abs(
                    reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]
                ).mean().item()
                watermark_decoded_log.append(w_metric)

                # ============================================================= #

                # Calculate the quality: mse and psnr
                mse_recon_w = calc_mse(im_w_bgr_uint8, img_bgr_uint8)
                psnr_recon_w = compute_psnr(
                    im_w_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
                )
                ssim_recon_w = compute_ssim(
                    im_w_bgr_uint8.astype(np.int16), img_bgr_uint8.astype(np.int16), data_range=255
                )

                
                mse_w_log.append(mse_recon_w)
                psnr_w_log.append(psnr_recon_w)
                ssim_w_log.append(ssim_recon_w)

            # Save the result
            processed_dict = {
                "index": index_log,
                "watermark_gt_str": watermark_gt_str, # Some historical none distructive bug :( will cause this reformatting
                "watermark_decoded": watermark_decoded_log,
                "psnr_w": psnr_w_log,
                "ssim_w": ssim_w_log
            }

            save_name = os.path.join(save_root_dir, file_name)
            with open(save_name, 'wb') as handle:
                pickle.dump(processed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Decoded Interm. result saved to {}".format(save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Some arguments to play with.')
    # ======
    parser.add_argument('--run_name', default='test')
    # parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    # parser.add_argument('--start', default=0, type=int)
    # parser.add_argument('--end', default=1000, type=int)
    # parser.add_argument('--end', default=2000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=50, type=int)
    parser.add_argument('--reference_model', default="ViT-g-14")
    parser.add_argument('--reference_model_pretrain', default="laion2b_s12b_b42k")
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)
    # ======

    parser.add_argument(
        "--watermarker", dest="watermarker", type=str, 
        help="Specification of watermarking method. [rivaGan, dwtDctSvd]",
        default="Tree-Ring"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str, 
        help="Dataset [COCO, DiffusionDB]",
        default="Gustavosta"
    )
    parser.add_argument(
        "--evade_method", dest="evade_method", type=str, help="Specification of evasion method.",
        default="vae"
    )
    parser.add_argument(
        "--arch", dest="arch", type=str, 
        help="""
            Secondary specification of evasion method (if there are other choices).

            Valid values a listed below:
                dip --- ["vanila", "random_projector"],
                vae --- ["cheng2020-anchor", "mbt2018", "bmshj2018-factorized"],
                corrupters --- ["gaussian_blur", "gaussian_noise", "bm3d", "jpeg", "brightness", "contrast"]
                diffuser --- Do not need.
        """,
        default="cheng2020-anchor"
    )
    parser.add_argument(
        "--start", dest="start", type=int, help="Specification of evasion method.",
        default=0
    )
    parser.add_argument(
        "--end", dest="end", type=int, help="Specification of evasion method.",
        default=2001
    )
    args = parser.parse_args()
    main(args)
    
    # root_lv1 = os.path.join("..", "DIP_Watermark_Evasion", "Result-Interm", args.watermarker, args.dataset)
    # corrupter_names = [f for f in os.listdir(root_lv1)]
    # for corrupter in corrupter_names:
    #     root_lv2 = os.path.join(root_lv1, corrupter)
    #     arch_names = [f for f in os.listdir(root_lv2)]
    #     for arch in arch_names:
    #         args.evade_method = corrupter
    #         args.arch = arch
    #         print("Processing: {} - {} - {} - {}".format(args.watermarker, args.dataset, args.evade_method, args.arch))
    #         main(args)
    print("\n***** Completed. *****\n")