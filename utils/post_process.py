from PIL import Image, ImageFile
from argparse import ArgumentParser
import os
import shutil
import torch
import torch.nn.functional as F
import torch.distributed as dist
import clip  # 也就是 openai-clip
import hpsv2 # [NEW] 引入 HPSv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 模型初始化 ---
def init_clip_model(device):
    # 使用 ViT-B/16 或 L/14 都可以，这里保持原设置
    print("Loading CLIP model for Image Consistency...")
    model, preprocess = clip.load('ViT-B/16', device)
    return model, preprocess

# --- [NEW] 核心评分逻辑 ---
def compute_scores(clip_model, preprocess, input_img_path, edit_img_dir, input_text, device):
    img_list = sorted(os.listdir(edit_img_dir))
    valid_img_paths = []
    valid_img_tensors = []
    
    # 1. 加载所有候选图
    for img_name in img_list:
        if not img_name.endswith(('.png', '.jpg', '.jpeg')): continue
        path = os.path.join(edit_img_dir, img_name)
        try:
            # 为 CLIP 准备 tensor
            pil_img = Image.open(path).convert('RGB')
            valid_img_tensors.append(preprocess(pil_img).unsqueeze(0))
            valid_img_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            pass

    if not valid_img_paths:
        return None, None, None

    # 2. 计算 HPSv2 分数 (替代 CLIP TTI)
    # 衡量：Text Alignment & Human Preference
    print("Calculating HPSv2 scores (Text Alignment)...")
    # hpsv2.score 会自动下载并加载模型，返回一个分数 list
    hps_scores_list = hpsv2.score(valid_img_paths, input_text, hps_version="v2.1")
    hps_scores = torch.tensor(hps_scores_list).to(device)

    # 3. 计算 CLIP Image-to-Image 相似度 (保留原 CLIP ITI)
    # 衡量：Semantic Fidelity (语义保真度)
    print("Calculating CLIP Image Similarity (Semantic Fidelity)...")
    
    src_img = preprocess(Image.open(input_img_path).convert('RGB')).unsqueeze(0).to(device)
    cand_imgs = torch.cat(valid_img_tensors, dim=0).to(device)
    
    with torch.no_grad():
        # 编码原图和候选图
        src_features = clip_model.encode_image(src_img)
        cand_features = clip_model.encode_image(cand_imgs)
        
        # 归一化特征
        src_features = src_features / src_features.norm(dim=-1, keepdim=True)
        cand_features = cand_features / cand_features.norm(dim=-1, keepdim=True)
        
        # 计算余弦相似度
        # shape: [num_images]
        clip_iti_scores = (cand_features @ src_features.T).squeeze(1)

    return hps_scores, clip_iti_scores, valid_img_paths

# --- 主流程 ---
def get_final_img(args, input_text, input_img_path, edit_img_path, topk_tti=5):
    # alpha: HPSv2 权重 (Text Alignment)
    # beta: CLIP-Image 权重 (Content Fidelity)
    alpha, beta = args.test_alpha, args.test_beta
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 初始化 CLIP
    clip_model, preprocess = init_clip_model(device)
    
    # 2. 计算分数
    # hps_scores -> 代表"听话程度"
    # clip_iti_scores -> 代表"像原图程度"
    hps_scores, clip_iti_scores, img_paths = compute_scores(
        clip_model, preprocess, input_img_path, edit_img_path, input_text, device
    )
    
    if hps_scores is None:
        print("No valid images found.")
        return

    # --- 3. 归一化 (Normalization) [关键步骤] ---
    # HPSv2 和 CLIP Sim 的数值范围不同，必须归一化到 0-1 之间才能加权
    def normalize(t):
        if t.max() == t.min(): return torch.zeros_like(t)
        return (t - t.min()) / (t.max() - t.min())

    norm_hps = normalize(hps_scores)
    norm_clip = normalize(clip_iti_scores)

    # --- 4. 筛选逻辑 (Ranking) ---
    
    # A. 找出 HPS 分数最高的 Top-K (最符合指令的一批)
    k = min(topk_tti, len(img_paths))
    _, indices_tti = torch.topk(hps_scores, k=k)
    indices_tti_set = set(indices_tti.tolist())

    # B. 计算综合分数
    # Formula: Score = alpha * HPS + beta * CLIP_Image
    final_score = alpha * norm_hps + beta * norm_clip
    
    # C. 对综合分数排序
    _, sorted_indices = torch.sort(final_score, descending=True)
    
    img_save = []
    
    # D. 双重验证筛选
    # 优先选择：综合分高 且 HPS也在前K名 的图
    for idx in sorted_indices:
        idx_val = idx.item()
        if idx_val in indices_tti_set:
            img_save.append(img_paths[idx_val])
            break # 找到第一名就停止
    
    # 如果交集为空，兜底使用 HPS 分数最高的那张
    if not img_save:
        best_hps_idx = indices_tti[0].item()
        img_save.append(img_paths[best_hps_idx])

    print(f"Selected Best Image: {os.path.basename(img_save[0])}")

    # --- 5. 保存结果 ---
    try:
        rank = dist.get_rank()
    except:
        rank = 0

    if rank == 0:
        output_name = 'final_output.png'
        output_full_path = os.path.join(edit_img_path, output_name)
        shutil.copyfile(img_save[0], output_full_path)
        print(f"Saved to {output_full_path}")

# 兼容命令行调用
if __name__ == "__main__":
    parser = ArgumentParser()
    # 建议权重：HPS稍微重要一点，因为它决定了编辑是否生效
    parser.add_argument('--test_alpha', type=float, default=0.6, help="Weight for HPSv2 (Text)")
    parser.add_argument('--test_beta', type=float, default=0.4, help="Weight for CLIP (Image)")
    args, _ = parser.parse_known_args()
    # 示例调用 (记得注释掉)
    # get_final_img(args, "text prompt", "input.jpg", "./outputs/")