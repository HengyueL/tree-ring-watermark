import argparse
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import pandas as pd
import torch
import PIL
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *


# def uint8_to_float(img_orig):
#     """
#         Convert a uint8 image into float with range [0, 1]
#     """
#     return img_orig.astype(np.float32) / 255.


# def img_np_to_tensor(img_np):
#     """
#         Convert numpy image (float with range [0, 1], shape (N, N, 3)) into tensor input with shape (1, 3, N, N)
#     """
#     img_np = np.transpose(img_np, [2, 0, 1])
#     img_np = img_np[np.newaxis, :, :, :]
#     img_tensor = torch.from_numpy(img_np)
#     return img_tensor


# def load_image(dir, idx):
#     image_path = os.path.join(
#         dir, "Img-{}.png".format(idx)
#     )
#     img_bgr_uint8 = cv2.imread(image_path)
#     img_bgr_float = uint8_to_float(img_bgr_uint8)
#     img_bgr_tensor = img_np_to_tensor(img_bgr_float)
#     return img_bgr_tensor


def main(args):
    dataset_root = os.path.join(
        ".", "dataset", "Tree-Ring", args.dataset.split("/")[0]
    )
    save_clean_image_root = os.path.join(
        dataset_root, "no_watermark_image"
    )
    save_watermarked_image_root = os.path.join(
        dataset_root, "encoder_image"
    )
    save_csv_dir = os.path.join(dataset_root, "water_mark.csv")
    res_dict = {
        "ImageName": [],
        "Decode_Clean": [],
        "Decode_W": [],
    }
    os.makedirs(save_clean_image_root, exist_ok=True)
    os.makedirs(save_watermarked_image_root, exist_ok=True)
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        print("Check clean image shape and type: ", type(orig_image_no_w))
        save_image_path = os.path.join(save_clean_image_root, "Img-{}.png".format(i))
        orig_image_no_w.save(save_image_path, "PNG")

        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]
        print("Check watermarked image shape and type: ", type(orig_image_w))
        save_image_path = os.path.join(save_watermarked_image_root, "Img-{}.png".format(i))
        orig_image_w.save(save_image_path, "PNG")

        ### test watermark
        # distortion
        # orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)
        orig_image_no_w_auged, orig_image_w_auged = orig_image_no_w, orig_image_w

        # reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse img with watermarking
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)
        
        print("No watermark image tree-ring metric value: ", no_w_metric)
        print("Watermarked image tree-ring metric value: ", w_metric)
        res_dict["ImageName"].append("Img-{}.png".format(i))
        res_dict["Decode_Clean"].append(no_w_metric)
        res_dict["Decode_W"].append(w_metric)

    df = pd.DataFrame(res_dict)
    df.to_csv(save_csv_dir, index=False)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    # parser.add_argument('--end', default=1000, type=int)
    parser.add_argument('--end', default=2000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
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

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)

