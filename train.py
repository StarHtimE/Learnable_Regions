from tqdm import tqdm
from einops import rearrange
from PIL import Image
from copy import deepcopy
from torchvision.utils import save_image
from typing import List, Optional, Union
from torch import autocast
from torchvision import utils as vutils
from utils.util import EditingJsonDataset, EditingSingleImageDataset, plot_images
from lr_schedule import WarmupLinearLRSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from models.model import RGN
from models.utils import visualize_images, read_image_from_url, draw_image_with_bbox_new, Bbox
from utils.util2 import compose_text_with_templates, get_augmentations_template
from torchvision.utils import draw_bounding_boxes
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from engine import *
from vis import *
import os, jax, cv2, pdb
import numpy as np
import argparse, torch, inspect
import PIL, time, json, datetime
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils.misc as misc
import torchvision.transforms as T
import torch.distributed as dist


def _maybe_dashscope_enrich(args, rank):
    """Optionally call DashScope to get image description and edited-image description."""
    if not (args.use_dashscope_caption or args.use_dashscope_edit_description):
        return args

    if args.json_file is not None:
        if rank == 0:
            print("DashScope enrichment currently supports single-image mode (image_file_path). Skipping.")
        return args

    api_key = args.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
    if api_key is None:
        raise RuntimeError("DashScope API key missing. Set --dashscope_api_key or DASHSCOPE_API_KEY.")

    caption = None
    edited_desc = None

    if rank == 0:
        try:
            import dashscope  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("dashscope is required. Install with: pip install dashscope") from exc

        # ------------------------------------------------------------------
        # 1) MLLM: image -> faithful source description (SDXL-compatible)
        # ------------------------------------------------------------------
        if args.use_dashscope_caption or args.image_caption is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": args.image_file_path},
                        {
                            "text": (
                                "Describe the image for use as a diffusion model prompt.\n"
                                "Rules:\n"
                                "- Describe only what is visible.\n"
                                "- Use concrete nouns and visual attributes.\n"
                                "- Include spatial relationships when relevant.\n"
                                "- Avoid interpretation or speculation.\n"
                                "- Do NOT mention editing, camera, or photography terms unless obvious.\n"
                                "- Keep it concise but complete.\n"
                            )
                        },
                    ],
                }
            ]
            resp = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=args.dashscope_vl_model,
                messages=messages,
            )
            caption = resp.output.choices[0].message.content[0]["text"]

        # ------------------------------------------------------------------
        # 2) LLM: (source description + edit instruction) -> target description
        # ------------------------------------------------------------------
        if args.use_dashscope_edit_description:
            base_caption = caption if caption is not None else args.image_caption
            if base_caption is None:
                raise RuntimeError("Cannot generate edited description without an image caption.")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate image descriptions for an SDXL inpainting model.\n"
                        "You act as a prompt compiler, not a conversational assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Source image description:\n"
                        f"{base_caption}\n\n"
                        "Edit instruction:\n"
                        f"{args.editing_prompt}\n\n"
                        "Task:\n"
                        "- Generate a new image description in one sentence that reflects the edit instruction.\n"
                        "- MUST output less than 15 words.\n"
                        "- ONLY describe objects changed, ignore details like light or environment.\n"
                        "- Do NOT explain the changes.\n"
                        "- Do NOT mention the original image or the editing process.\n"
                        "- Use concrete, visual language compatible with SDXL.\n"
                        "- Maintain original style and lighting unless instructed otherwise.\n"
                        "\nOutput only the final image description."
                    ),
                },
            ]

            resp = dashscope.Generation.call(
                api_key=api_key,
                model=args.dashscope_text_model,
                messages=messages,
                result_format="message",
            )
            if resp.status_code == 200:
                edited_desc = resp.output.choices[0].message.content
            else:
                raise RuntimeError(
                    f"DashScope Generation failed: status={resp.status_code}, "
                    f"code={resp.code}, message={resp.message}"
                )

    # ----------------------------------------------------------------------
    # Distributed sync
    # ----------------------------------------------------------------------
    shared = [caption, edited_desc]
    dist.broadcast_object_list(shared, src=0)
    caption, edited_desc = shared


    if caption is not None and (args.image_caption is None or args.override_image_caption):
        args.image_caption = " ".join(caption.split()[:5])
    args.editing_prompt = edited_desc

    if rank == 0:
        if caption is not None:
            print(f"[DashScope] image_description: {caption}")
        if edited_desc is not None:
            print(f"[DashScope] edited_image_description: {edited_desc}")

    return args


def configure_optimizers(model, lr, betas=(0.9, 0.96), weight_decay=4.5e-2):
    optimizer = torch.optim.Adam(model.module.anchor_net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    return optimizer

def train(args, lr_schedule, model, template, len_train_dataset, data_loader_train, optim, device_id):
    save_path = args.save_path
    rank = dist.get_rank()
    if not os.path.exists(save_path) and rank == 0:
        os.mkdir(save_path)

    for epoch in range(1, args.epochs+1):
        data_loader_train.sampler.set_epoch(epoch)
        metric_logger = misc.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if rank == 0:
            print(f'Epoch {epoch}:')
        for data_iter_step, (imgs, o_prompt, e_prompt) in enumerate(tqdm(data_loader_train)):
            lr_schedule.step()
            imgs = imgs.to(device=device_id, non_blocking=True)
            o_prompt, e_prompt = o_prompt[0], e_prompt[0]
            e_prompt = compose_text_with_templates(e_prompt, template) 
            with torch.cuda.amp.autocast():
                bboxs = torch.ceil(map_cooridates(model.module.get_anchor_box(imgs)))
                imgs_new, mask_imgs = get_mask_imgs(imgs, bboxs)
                results = model.module.generate_result(imgs_new.to(device_id), mask_imgs.to(device_id), e_prompt).to(device_id)
                loss, loss_clip, loss_cip_dir, loss_structure = model.module.get_loss(imgs_new, results, e_prompt, o_prompt)
            loss.backward()
            if data_iter_step % args.accum_grad == 0:
                optim.step()
                optim.zero_grad()
            metric_logger.update(loss=loss.item())

        if rank == 0:
            if epoch % args.ckpt_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_path, f'transformer_epoch_{epoch}.pth'))
            torch.save(model.state_dict(), os.path.join(save_path, 'last.pth'))

    return model

def main(args):
    dist.init_process_group("nccl", init_method='env://')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark= True

    template = get_augmentations_template()

    args = _maybe_dashscope_enrich(args, rank)
    
    if not os.path.exists(args.save_path) and rank == 0:
        os.mkdir(args.save_path)

    model = RGN(image_size=args.image_size, device=device_id, args=args).to(device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    if rank == 0 and not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)    
    dirs = os.listdir(args.output_dir)
    
    if args.json_file is not None:
        train_dataset = EditingJsonDataset(args, args.per_image_iteration)
        test_dataset = EditingJsonDataset(args)
    else:
        train_dataset = EditingSingleImageDataset(args, args.per_image_iteration)
        test_dataset = EditingSingleImageDataset(args)
    len_train_dataset = len(train_dataset)

    num_tasks = misc.get_world_size()
    sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=rank, shuffle=False, drop_last=False
        )

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, 
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_mem)

    optim = configure_optimizers(model, args.lr)
    total_steps = len_train_dataset / (args.batch_size*num_tasks)
    lr_schedule = CosineAnnealingLR(optim, T_max=args.epochs*total_steps)
    optim.zero_grad()
    model = train(args, lr_schedule, model, template, len_train_dataset, data_loader_train, optim, device_id)
    if rank == 0:
        print('Generating edited images!')
        model.eval()
        predict(args, model, template, data_loader_test, device_id)


def get_args_parser():
    parser = argparse.ArgumentParser(description="train models")
    parser.add_argument('--run_name', type=str, default="exp")
    parser.add_argument("--nodes", default=1, type=int, help='number of nodes to request')
    parser.add_argument('--image_size', type=int, default=256, help='image height and width.')
    parser.add_argument('--image_dir_path', type=str, default=None, help='dir path to input images.')
    parser.add_argument('--image_file_path', type=str, default=None, help='path to input images.')
    parser.add_argument('--image_caption', type=str, default=None, help='caption of the input image.')
    parser.add_argument('--editing_prompt', type=str, default=None, help='editing prompt.')
    parser.add_argument('--json_file', type=str, default=None, help='path to image-prompt file.')
    parser.add_argument('--draw_box', action='store_true', help='draw boxes')
    parser.add_argument('--diffusion_model_path', type=str, default='runwayml/stable-diffusion-inpainting', help='path to stable diffusion model.')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='path to save checkpoint.')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='path to save checkpoint.')
    parser.add_argument('--output_dir', type=str, default='./output', help='path to output dir.')
    parser.add_argument('--device', type=str, default="cuda", help='device the training is on.')
    parser.add_argument('--batch_size', type=int, default=192, help='batch size for training.')
    parser.add_argument('--accum_grad', type=int, default=25, help='number for gradient accumulation.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train.')
    parser.add_argument('--per_image_iteration', type=int, default=1, help='training iterations for each image.')
    parser.add_argument('--loss_alpha', type=int, default=1, help='coefficient of clip loss.')
    parser.add_argument('--loss_beta', type=int, default=1, help='coefficient of directional clip loss.')
    parser.add_argument('--loss_gamma', type=int, default=1, help='coefficient of sturcture loss.')
    parser.add_argument('--test_alpha', type=int, default=2, help='coefficient of text-to-image similarity.')
    parser.add_argument('--test_beta', type=int, default=1, help='coefficient of image-to-image similarity.')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='number of epochs to save.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate.')
    parser.add_argument('--max_window_size', type=int, default=6, help='max window size')
    parser.add_argument('--point_number', type=int, default=3, help='point sample number')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--save_attn_viz', action='store_true', help='Save attention heatmaps and anchor overlays during prediction (rank0 only).')
    # dashscope-assisted captioning / description
    parser.add_argument('--use_dashscope_caption', action='store_true', help='Use DashScope qwen-vl to auto caption the input image (single-image mode).')
    parser.add_argument('--use_dashscope_edit_description', action='store_true', help='Use DashScope qwen-plus to produce edited image description (logs only).')
    parser.add_argument('--override_image_caption', action='store_true', help='Allow DashScope caption to overwrite provided image_caption.')
    parser.add_argument('--dashscope_api_key', type=str, default=None, help='DashScope API key; otherwise read DASHSCOPE_API_KEY env.')
    parser.add_argument('--dashscope_vl_model', type=str, default='qwen3-vl-plus', help='DashScope VL model id.')
    parser.add_argument('--dashscope_text_model', type=str, default='qwen-plus', help='DashScope text model id.')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args_parser()
    main(args)
    
