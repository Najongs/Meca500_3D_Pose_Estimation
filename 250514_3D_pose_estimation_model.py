import os
import re
import math
import json
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer

from ultralytics import YOLO

# ─── 1) 유틸 함수들 (변화 없음) ──────────────────────────────────────────────────────

def calculate_coordinate_mse(model, dataloader, device='cpu'):
    """
    모델의 3D 좌표 예측에 대한 MSE를 계산합니다.
    """
    model.eval()
    total_mse = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    count = 0

    with torch.no_grad():
        for img_batch, keypoint_batch in dataloader:
            img_batch = img_batch.to(device, non_blocking=True)
            keypoint_batch = keypoint_batch.to(device, non_blocking=True)
            preds = model(img_batch)

            pred_np = preds.cpu().numpy()
            gt_np = keypoint_batch.cpu().numpy()

            # Add a shape check for robustness
            if pred_np.shape != gt_np.shape:
                print(f"Warning: Prediction shape {pred_np.shape} does not match Ground Truth shape {gt_np.shape}. Skipping batch for MSE calculation.")
                continue

            mse_x = ((gt_np[..., 0] - pred_np[..., 0]) ** 2).mean()
            mse_y = ((gt_np[..., 1] - pred_np[..., 1]) ** 2).mean()
            mse_z = ((gt_np[..., 2] - pred_np[..., 2]) ** 2).mean()

            total_mse['x'] += mse_x
            total_mse['y'] += mse_y
            total_mse['z'] += mse_z
            count += 1

    avg_mse_x = total_mse['x'] / count if count else 0.0
    avg_mse_y = total_mse['y'] / count if count else 0.0
    avg_mse_z = total_mse['z'] / count if count else 0.0
    avg_mse_total = (avg_mse_x + avg_mse_y + avg_mse_z) / 3

    return avg_mse_x, avg_mse_y, avg_mse_z, avg_mse_total

def save_plot(data, title, fname, ylim=None, save_dir='./results'): # Added save_dir argument
        plt.figure()
        plt.plot(data, marker='o')
        plt.title(title)
        plt.xlabel('Epoch (×10)')
        plt.ylabel('Value')
        if ylim is not None:
            plt.ylim(0, ylim)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, label_dir, yolo_model_path, yolo_model_device):
        self.label_dir = label_dir
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")],
                                  key=lambda x: int(re.search(r'\d+', x).group()))

        self.data = []
        for label_fname in self.label_files:
            label_path = os.path.join(self.label_dir, label_fname)
            with open(label_path, 'r') as f:
                label_data = json.load(f)
                img_path_from_json = label_data["image_path"]

                if not os.path.exists(img_path_from_json):
                    print(f"Warning: Image file not found at path specified in JSON: {img_path_from_json}. Skipping this data point.")
                    continue

                # IMPORTANT: Ensure joint_coords_camera_frame_meters has 7 points
                joint_coords = np.array(label_data["joint_coords_camera_frame_meters"], dtype=np.float32)
                if joint_coords.shape[0] != 7:
                    print(f"Warning: Expected 7 keypoints but found {joint_coords.shape[0]} in {img_path_from_json}. Skipping.")
                    continue

                self.data.append({
                    "image_path": img_path_from_json,
                    "joint_coords": joint_coords
                })

        if not self.data:
            raise RuntimeError(f"No valid image-label pairs found in {label_dir}. Check paths and file existence.")

        # ViT는 ImageNet 사전학습 시 224x224를 많이 사용합니다.
        # 일반적인 ViT는 정규화 스케일이 ImageNet과 동일하게 맞춰져야 합니다.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # YOLO 모델은 여기서 초기화하지 않고, 메인 학습 스크립트에서만 사용하도록 분리하는 것이 좋습니다.
        # Dataset은 이미지와 라벨 데이터를 로드하는 역할에 집중하고,
        # YOLO 추론은 DataLoader에서 배치 단위로 미리 처리하거나, 학습 루프 바깥에서 한 번에 처리하는 것이 효율적입니다.
        # 현재 코드에서는 __getitem__에서 YOLO를 사용하므로 유지하되, 이로 인해 속도가 느려질 수 있음을 인지해야 합니다.
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)
            self.yolo_model.to(yolo_model_device)
            print(f"YOLO model loaded on: {yolo_model_device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Proceeding without YOLO. Ensure bounding boxes are correctly provided elsewhere if needed.")
            self.yolo_model = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        img_path = data_point["image_path"]
        target_coords = data_point["joint_coords"] # 7 keypoints

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}. Returning dummy data.")
            dummy_img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_target = torch.zeros((7, 3), dtype=torch.float32)
            return dummy_img_tensor, dummy_target

        cropped_img = img # Default to full image if YOLO fails or is not used

        if self.yolo_model:
            results = self.yolo_model.predict(img_path, imgsz=640, verbose=False)[0]

            if len(results.boxes.xyxy) == 0:
                print(f"Warning: YOLO failed to detect robot in {img_path}. Using full image.")
                # Fallback to full image
            else:
                x1, y1, x2, y2 = map(int, results.boxes.xyxy[0].cpu().numpy())
                h_orig, w_orig, _ = img.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_orig, x2)
                y2 = min(h_orig, y2)

                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid bounding box after clipping for {img_path}. Using full image. Bbox: ({x1},{y1},{x2},{y2})")
                    cropped_img = img
                else:
                    cropped_img = img[y1:y2, x1:x2]
        else:
            print(f"Warning: YOLO model not loaded. Using full image for {img_path}.")


        resized_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        tensor_img = self.transform(rgb_img)

        target = torch.from_numpy(target_coords).float() # 7 keypoints
        
        return tensor_img, target

# ─── 3) Model (HRNet-like with ViT backbone, 7 keypoints) ─────────────────────────────

# 기존 BasicBlock은 CNN 기반 보조 브랜치 또는 헤드에 사용될 수 있습니다.
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.down  = (nn.Conv2d(in_c,out_c,1,stride,bias=False)
                      if (stride!=1 or in_c!=out_c) else None)
    def forward(self,x):
        res = self.down(x) if self.down else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out+res)

# ViT 출력을 공간 특징 맵으로 재구성하는 유틸리티 함수
def reshape_vit_output(x, patch_size=16):
    # x: (B, num_patches + 1, dim) -> (B, num_patches, dim) (CLS 토큰 제거)
    # num_patches = H_img / patch_size * W_img / patch_size
    # dim: ViT의 임베딩 차원
    
    if x.ndim == 3 and x.shape[1] > 1: # Batch, Sequence Length, Embedding Dim
        # Remove CLS token if present
        if x.shape[1] == (224 // patch_size)**2 + 1: # Assuming 224x224 input
            x = x[:, 1:] 
        
        B, N, D = x.shape
        # Assuming square patches and image, N = H_feat * W_feat
        H_feat = W_feat = int(N**0.5)
        if H_feat * W_feat != N:
            raise ValueError(f"ViT output sequence length {N} is not a perfect square (after CLS token removal). Cannot reshape to square feature map.")
        
        # Reshape to (B, D, H_feat, W_feat)
        x = x.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
    
    elif x.ndim == 4: # Already CNN-like (B, C, H, W)
        pass # Do nothing
    else:
        raise ValueError(f"Unsupported ViT output shape for reshaping: {x.shape}")
    
    return x

# 메인 모델
class PoseEstimationHRViT(nn.Module):
    def __init__(self, num_kp=7, vit_model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.num_kp = num_kp

        # 1. ViT Backbone (고해상도 특징 추출)
        # timm 라이브러리에서 사전 학습된 ViT 모델 로드
        # out_indices를 사용하여 중간 레이어의 특징을 가져올 수도 있지만,
        # 여기서는 마지막 레이어의 출력을 사용하고 CNN으로 추가 처리합니다.
        self.vit_backbone = VisionTransformer(
            img_size=224, patch_size=16, in_chans=3, num_classes=0, # num_classes=0으로 설정하여 분류 헤드 제거
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
            norm_layer=nn.LayerNorm
        )
        if pretrained:
            # timm에서 제공하는 사전 학습 가중치 로드
            import timm
            pretrained_vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0)
            self.vit_backbone.load_state_dict(pretrained_vit.state_dict())
            print(f"Loaded pretrained ViT model: {vit_model_name}")

        # ViT의 임베딩 차원 (예: vit_base_patch16_224의 경우 768)
        vit_embed_dim = self.vit_backbone.embed_dim 
        vit_patch_size = self.vit_backbone.patch_embed.patch_size[0]

        # ViT 출력의 공간 해상도 계산 (224x224 입력, 16x16 패치 -> 14x14 패치 그리드)
        vit_feat_H = 224 // vit_patch_size
        vit_feat_W = 224 // vit_patch_size
        
        # 2. ViT 출력 후처리 및 HRNet 유사 구조 (고해상도 브랜치)
        # ViT 출력을 공간 특징 맵 (B, D, H_feat, W_feat) 형태로 변환
        # D는 vit_embed_dim, H_feat, W_feat는 각각 14
        
        # HRNet의 Stage1에 해당하는 고해상도 처리
        # ViT의 출력 채널을 CNN에 적합하게 조정
        self.high_res_branch_conv = nn.Sequential(
            nn.Conv2d(vit_embed_dim, 256, 1, bias=False), # 채널 조정
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 추가적인 BasicBlock 레이어들을 여기에 추가하여 고해상도 브랜치 강화
            self._make_layer(256, 256, 2, 1) # 해상도 유지
        )
        
        # 3. 저해상도 브랜치 (CNN 기반)
        # 고해상도 브랜치에서 다운샘플링하여 시작
        self.low_res_branch_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), # 해상도 절반으로 줄임 (14x14 -> 7x7)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            self._make_layer(512, 512, 2, 1) # 해상도 유지
        )

        # 4. HRNet 유사 융합 모듈
        # 고해상도 브랜치와 저해상도 브랜치 간의 정보 교환
        # Stage 2 (고해상도 (256ch, 14x14) <-> 저해상도 (512ch, 7x7))
        self.fuse_h_to_l = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fuse_l_to_h = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False), # 채널 조정
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest') # 업샘플링 (7x7 -> 14x14)
        )
        
        # 5. 최종 회귀 헤드
        # HRNet처럼 최종 고해상도 브랜치에서 키포인트 예측
        # 또는 모든 해상도 브랜치를 최종적으로 융합한 후 예측
        # 여기서는 가장 간단하게 고해상도 브랜치 출력을 사용합니다.
        
        # 최종 특징 맵 크기 (예: 256ch, 14x14)
        # Global Average Pooling 후 Linear
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 128, 1), # 채널 축소
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # (B, 128, 1, 1)
            nn.Flatten(), # (B, 128)
            nn.Linear(128, num_kp * 3) # 7 * 3 = 21 출력
        )

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. ViT Backbone
        # x: (B, 3, 224, 224)
        vit_output = self.vit_backbone.forward_features(x) # (B, num_patches + 1, embed_dim)
        
        # 2. ViT 출력을 공간 특징 맵으로 재구성
        # (B, embed_dim, H_feat, W_feat)
        # 예: (B, 768, 14, 14)
        vit_spatial_features = reshape_vit_output(vit_output, self.vit_backbone.patch_embed.patch_size[0])

        # 3. 고해상도 브랜치 처리 (ViT 특징 기반)
        high_res_feat = self.high_res_branch_conv(vit_spatial_features) # (B, 256, 14, 14)

        # 4. 저해상도 브랜치 처리
        low_res_feat = self.low_res_branch_conv(high_res_feat) # (B, 512, 7, 7)

        # 5. HRNet 유사 융합
        # 고해상도 브랜치에 저해상도 브랜치 정보 추가
        high_res_fused = high_res_feat + self.fuse_l_to_h(low_res_feat)
        # 저해상도 브랜치에 고해상도 브랜치 정보 추가 (옵션, 여기서는 최종 헤드에 사용 안함)
        # low_res_fused = low_res_feat + self.fuse_h_to_l(high_res_feat)

        # 6. 최종 회귀 헤드
        # 고해상도 융합 특징을 사용하여 키포인트 예측
        output = self.regression_head(high_res_fused)
        
        # 출력 형태를 (Batch, num_kp, 3)으로 재구성
        return output.view(-1, self.num_kp, 3)

# ─── 4) 각 GPU 프로세스별 훈련 함수 ────────────────────────────────────

def train_worker(rank, world_size, args):
    # 분산 설정
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']='29500'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank); device = torch.device(f'cuda:{rank}')

    if rank == 0:
        wandb.init(project='meca500-3d-coord-yolo-live', config=args)

    yolo_model_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    full_dataset = PoseDataset(args['label_dir'], args['yolo_model_path'], yolo_model_device)
    n_train = int(0.95 * len(full_dataset))
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.get('seed', 42))
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(
        train_ds,
        batch_size=args['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- IMPORTANT CHANGE: Model initialized for 6 keypoints ---
    model     = PoseEstimationHRViT().to(device)
    model     = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    criterion = nn.SmoothL1Loss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    loss_history=[]; mse_x_history=[]; mse_y_history=[]; mse_z_history=[]; mse_total_history=[]
    os.makedirs(args['save_dir'], exist_ok=True)

    for epoch in range(args['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for imgs, tars in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            tars = tars.to(device, non_blocking=True)
            pred = model(imgs)
            loss = criterion(pred, tars)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        avg_loss = total / len(train_loader)

        scheduler.step()

        if rank == 0:
            loss_history.append(avg_loss)
            mse_x, mse_y, mse_z, mse_total = calculate_coordinate_mse(model, val_loader, device)
            mse_x_history.append(mse_x); mse_y_history.append(mse_y); mse_z_history.append(mse_z); mse_total_history.append(mse_total)

            wandb.log({
                'epoch': epoch+1,
                'train_loss': avg_loss,
                'val_mse_x': mse_x,
                'val_mse_y': mse_y,
                'val_mse_z': mse_z,
                'val_mse_total': mse_total,
                'lr': optimizer.param_groups[0]['lr']
            }, step=epoch+1)

            if (epoch + 1) % 10 == 0:
                ckpt_name = f"epoch_{epoch+1:03d}_loss_{avg_loss:.4f}_mse_val_{mse_total:.4f}.pth"
                path = os.path.join(args['save_dir'], ckpt_name)
                # Save only the state_dict of the underlying model (not DDP wrapper)
                torch.save(model.module.state_dict(), path) 

                print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}, mse_total={mse_total:.4f} → saved {ckpt_name}")
                print(f"[Epoch {epoch+1}] mse_x_val={mse_x:.4f}, mse_y_val={mse_y:.4f}, mse_z_val={mse_z:.4f}")

            else:
                print(f"[Epoch {epoch+1}] loss={avg_loss:.4f}, mse_total={mse_total:.4f}")
                print(f"[Epoch {epoch+1}] mse_x_val={mse_x:.4f}, mse_y_val={mse_y:.4f}, mse_z_val={mse_z:.4f}")

    if rank == 0:
        save_plot(loss_history, 'Train Loss', 'train_loss.png', ylim=2, save_dir=args['save_dir'])
        save_plot(mse_x_history, 'Validation MSE - X', 'val_mse_x.png', ylim=0.01, save_dir=args['save_dir'])
        save_plot(mse_y_history, 'Validation MSE - Y', 'val_mse_y.png', ylim=0.01, save_dir=args['save_dir'])
        save_plot(mse_z_history, 'Validation MSE - Z', 'val_mse_z.png', ylim=0.01, save_dir=args['save_dir'])
        save_plot(mse_total_history, 'Validation MSE - Total', 'val_mse_total.png', ylim=0.01, save_dir=args['save_dir'])
        wandb.finish()

    dist.destroy_process_group()

# ─── 5) 엔트리포인트 ─────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = {
        'label_dir' : "/home/najo/NAS/Meca500_3D_Pose_Estimation/3D_Coordinate_label/left",
        'yolo_model_path': "/home/najo/NAS/Meca500_3D_Pose_Estimation/runs/detect/custom_yolov11X_full_train/weights/best.pt",
        'batch_size': 50,
        'epochs'    : 200,
        'save_dir'  : "/home/najo/NAS/Meca500_3D_Pose_Estimation/model_save/6kp_model" # Changed save directory for 6 keypoints
    }

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training.")
    mp.spawn(train_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)