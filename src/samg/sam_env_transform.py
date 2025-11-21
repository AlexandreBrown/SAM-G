import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import cv2
from segdac.data.mdp import MdpData
from segdac_dev.envs.transforms.transform import Transform
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    
    return topk_xy, topk_label


def negative_point_selection(mask_sim, topk=1, box=None):
	if box is None:
		box = np.array([0, 0, mask_sim.shape[0]-1, mask_sim.shape[1]-1])
	
	n_mask_sim = mask_sim.clone()
	n_mask_sim = n_mask_sim[box[1]:box[3], box[0]:box[2]]
	if n_mask_sim.shape[0] == 0 or n_mask_sim.shape[1] == 0:
		return np.array([[0, 0]]), np.array([0])
	n_topk_xy = n_mask_sim.flatten(0).topk(topk, largest=False)[1]
	w, h = n_mask_sim.shape
	n_topk_x = (n_topk_xy // h).unsqueeze(0)
	n_topk_y = (n_topk_xy - n_topk_x * h)
	n_topk_x = n_topk_x + box[1]
	n_topk_y = n_topk_y + box[0]
	n_topk_xy = torch.cat((n_topk_y, n_topk_x), dim=0).permute(1, 0)
	n_topk_label = np.array([0] * topk)
	n_topk_xy = n_topk_xy.cpu().numpy()

	return n_topk_xy, n_topk_label

class SamEnvTransform(Transform):
    def __init__(
        self, device: str,
        in_key: str,
        out_key: str,
        efficient_vit_model_name: str,
        efficient_vit_weights_path: str,
        original_image_path: str,
        masked_image_path: str,
        extra_points_list: list = [],
        extra_masked_images_list: list = []
    ):
        super().__init__(device)
        self.in_key = in_key
        self.out_key = out_key
        efficientvit_sam = create_sam_model(
            name=efficient_vit_model_name,
            weight_url=efficient_vit_weights_path
        )
        efficientvit_sam = efficientvit_sam.cuda().eval()
        self.efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

        print("loading dino model")
        # self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        # if due to some reasons, the above command doesn't work, you can load the model from a local path
        # you need to put the downloaded checkpoint in your ~/.cache/torch/hub/checkpoints/ directory
        # you should clone the dinov2 repo and put the path to the local directory (similar as efficientvit)
        # self.dino_model = torch.hub._load_local('../../../dinov2', 'dinov2_vitb14')
        
        diov2_vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        self.dino_model = diov2_vit.cuda().eval()
        print("dino model loaded")

        self.dino_transform = T.Compose([T.ToTensor(),
                             T.Resize(448),
                             T.Normalize(
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                             ),
                             ])
        self.dino_model.eval()
        
        target_feat_list = []
        dino_target_feat_list = []
        ref_image = cv2.imread(original_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        ref_mask = cv2.imread(masked_image_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        dino_ref_image = ref_image.copy()
        dino_ref_image = self.dino_transform(dino_ref_image).unsqueeze(0).cuda()
        dino_ref_image_embedding = self.dino_model.forward_features(dino_ref_image)
        patch_tokens = dino_ref_image_embedding["x_norm_patchtokens"]
        patch_tokens = patch_tokens.reshape([1, 32, 32, 768])
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)
        patch_tokens = F.interpolate(patch_tokens, size=(64, 64), mode='bilinear', align_corners=False)
        patch_tokens = patch_tokens.permute(0, 2, 3, 1)
        
        dino_ref_feat = patch_tokens.squeeze(0) # [64, 64, 768]
        gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

        ref_mask = self.efficientvit_sam_predictor.Per_set_image(ref_image, ref_mask)
        ref_feat = self.efficientvit_sam_predictor.features.squeeze(0).permute(1, 2, 0)
        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[:2], mode='bilinear', align_corners=False) 
        ref_mask = ref_mask.squeeze()[0]

        target_feat = ref_feat[ref_mask > 0]
        target_feat_mean = target_feat.mean(0)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        target_feat_list.append(target_feat)

        dino_target_feat = dino_ref_feat[ref_mask > 0]
        dino_target_feat_mean = dino_target_feat.mean(0)
        dino_target_feat_max = torch.max(dino_target_feat, dim=0)[0]
        dino_target_feat = (dino_target_feat_max / 2 + dino_target_feat_mean / 2).unsqueeze(0)
        dino_target_feat = dino_target_feat / dino_target_feat.norm(dim=-1, keepdim=True)
        dino_target_feat_list.append(dino_target_feat)

        h, w, C = ref_feat.shape
        ref_feat_ = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat_ = ref_feat_.permute(2, 0, 1).reshape(C, -1)
        sim = target_feat @ ref_feat_
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
        sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
            sim,
            input_size=self.efficientvit_sam_predictor.input_size,
            original_size=self.efficientvit_sam_predictor.original_size,
        ).squeeze()

        h, w, C = dino_ref_feat.shape
        dino_ref_feat_ = dino_ref_feat / dino_ref_feat.norm(dim=-1, keepdim=True)
        dino_ref_feat_ = dino_ref_feat_.permute(2, 0, 1).reshape(C, -1)
        dino_sim = dino_target_feat @ dino_ref_feat_
        dino_sim = dino_sim.reshape(1, 1, h, w)
        dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
        dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
            dino_sim,
            input_size=self.efficientvit_sam_predictor.input_size,
            original_size=self.efficientvit_sam_predictor.original_size,
        ).squeeze()

        # combine sim and dino_sim
        sim = (dino_sim + sim) / 2

        topk_xy, topk_label = point_selection(sim, topk=1)

        n_xy, n_label = negative_point_selection(sim, topk=1)
        topk_xy = np.concatenate((topk_xy, n_xy), axis=0)
        topk_label = np.concatenate((topk_label, n_label), axis=0)

        parts_target_feat = []
        dino_parts_target_feat = []
        for point in extra_points_list:
            y, x = point
            x = x / 84. * 64
            y = y / 84. * 64
            x1 = int(x); x2 = int(x) + 1
            y1 = int(y); y2 = int(y) + 1
            feat = ref_feat[x1, y1] * (x2 - x) * (y2 - y) \
                + ref_feat[x1, y2] * (x2 - x) * (y - y1) \
                + ref_feat[x2, y1] * (x - x1) * (y2 - y) \
                + ref_feat[x2, y2] * (x - x1) * (y - y1)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            parts_target_feat.append(feat)

            feat = dino_ref_feat[x1, y1] * (x2 - x) * (y2 - y) \
                + dino_ref_feat[x1, y2] * (x2 - x) * (y - y1) \
                + dino_ref_feat[x2, y1] * (x - x1) * (y2 - y) \
                + dino_ref_feat[x2, y2] * (x - x1) * (y - y1)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            dino_parts_target_feat.append(feat)

        for i in extra_masked_images_list:
            ref_mask = cv2.imread(i)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
            ref_mask = self.efficientvit_sam_predictor.Per_set_image(ref_image, ref_mask)
            ref_feat = self.efficientvit_sam_predictor.features.squeeze(0).permute(1, 2, 0)
            ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[:2], mode='bilinear', align_corners=False)
            ref_mask = ref_mask.squeeze()[0]

            target_feat = ref_feat[ref_mask > 0]
            target_feat_mean = target_feat.mean(0)
            target_feat_max = torch.max(target_feat, dim=0)[0]
            target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
            target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
            target_feat_list.append(target_feat)

            h, w, C = ref_feat.shape
            ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
            ref_feat = ref_feat.permute(2, 0, 1).reshape(C, -1)
            sim = target_feat @ ref_feat

            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
            sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
                sim,
                input_size=self.efficientvit_sam_predictor.input_size,
                original_size=self.efficientvit_sam_predictor.original_size,
            ).squeeze()
            
            dino_target_feat = dino_ref_feat[ref_mask > 0]
            dino_target_feat_mean = dino_target_feat.mean(0)
            dino_target_feat_max = torch.max(dino_target_feat, dim=0)[0]
            dino_target_feat = (dino_target_feat_max / 2 + dino_target_feat_mean / 2).unsqueeze(0)
            dino_target_feat = dino_target_feat / dino_target_feat.norm(dim=-1, keepdim=True)
            dino_target_feat_list.append(dino_target_feat)

            h, w, C = dino_ref_feat.shape
            dino_ref_feat_ = dino_ref_feat / dino_ref_feat.norm(dim=-1, keepdim=True)
            dino_ref_feat_ = dino_ref_feat_.permute(2, 0, 1).reshape(C, -1)
            dino_sim = dino_target_feat @ dino_ref_feat_
            dino_sim = dino_sim.reshape(1, 1, h, w)
            dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
            dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
                dino_sim,
                input_size=self.efficientvit_sam_predictor.input_size,
                original_size=self.efficientvit_sam_predictor.original_size,
            ).squeeze()

            sim = (dino_sim + sim) / 2
            topk_xy_, topk_label_ = point_selection(sim, topk=1)
            topk_xy = np.concatenate((topk_xy, topk_xy_), axis=0)
            topk_label = np.concatenate((topk_label, topk_label_), axis=0)

        mask_weights = Mask_Weights().cuda()
        mask_weights.train()

        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=1e-3, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

        for train_idx in range(1000):
            xy = topk_xy.copy()
            label = topk_label.copy()

            for i in range(len(dino_parts_target_feat)):
                part_feat = parts_target_feat[i]
                noise = torch.randn_like(part_feat)
                noise = noise / noise.norm(dim=-1, keepdim=True)
                part_feat = part_feat + noise * 0.01
                part_feat = part_feat / part_feat.norm(dim=-1, keepdim=True)
                sim = part_feat @ ref_feat_
                sim = sim.reshape(1, 1, h, w)
                sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
                sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
                    sim,
                    input_size=self.efficientvit_sam_predictor.input_size,
                    original_size=self.efficientvit_sam_predictor.original_size,
                ).squeeze()

                dino_target_feat = dino_parts_target_feat[i]
                noise = torch.randn_like(dino_target_feat)
                noise = noise / noise.norm(dim=-1, keepdim=True)
                dino_target_feat = dino_target_feat + noise * 0.01
                dino_target_feat = dino_target_feat / dino_target_feat.norm(dim=-1, keepdim=True)
                dino_sim = dino_target_feat @ dino_ref_feat_
                dino_sim = dino_sim.reshape(1, 1, h, w)
                dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
                dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
                    dino_sim,
                    input_size=self.efficientvit_sam_predictor.input_size,
                    original_size=self.efficientvit_sam_predictor.original_size,
                ).squeeze()

                sim = (dino_sim + sim) / 2
                xy_, label_ = point_selection(sim, topk=1)
                xy = np.concatenate((xy, xy_), axis=0)
                label = np.concatenate((label, label_), axis=0)
            masks, scores, logits, logits_high = self.efficientvit_sam_predictor.predict(
            point_coords=xy,
            point_labels=label,
            multimask_output=True)
            logits_high = logits_high.flatten(1).clone()

            # Weighted sum three-scale masks
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)

            dice_loss = calculate_dice_loss(logits_high, gt_mask)
            focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print("Finish training mask weights")

        self.weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        self.weights_np = self.weights.detach().cpu().numpy()

        self.target_feat = target_feat_list[0]
        self.dino_target_feat = dino_target_feat_list[0]
        if len(target_feat_list) > 1:
            self.part_target_feat = target_feat_list[1:] + parts_target_feat
            self.dino_parts_target_feat = dino_target_feat_list[1:] + dino_parts_target_feat
        else:
            self.part_target_feat = parts_target_feat
            self.dino_parts_target_feat = dino_parts_target_feat

    def reset(self, mdp_data: MdpData) -> MdpData:
        images = mdp_data.data[self.in_key].squeeze(1).clone()  # (b,c,h,w)
        mdp_data.data[self.out_key] = self._extract_pixels(images).unsqueeze(0).unsqueeze(0) # (1, 1, c, h, w) because SAM-G only supports 1 env, second 1 is timestep axis
        return mdp_data
    
    def _extract_pixels(self, pixels):
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        pixels = pixels.cpu().numpy() # (c,h,w)
        obs = pixels.copy()
        obs_feat = obs.copy()
        dino_obs_feat = self.dino_transform(obs_feat.transpose(1, 2, 0)).unsqueeze(0).cuda()
        dino_obs_feat = self.dino_model.forward_features(dino_obs_feat)
        patch_tokens = dino_obs_feat["x_norm_patchtokens"]
        patch_tokens = patch_tokens.reshape([1, 32, 32, 768])
        patch_tokens = patch_tokens.permute(0, 3, 1, 2)
        patch_tokens = F.interpolate(patch_tokens, size=(64, 64), mode='bilinear', align_corners=False)
        patch_tokens = patch_tokens.permute(0, 2, 3, 1)
        
        dino_obs_feat = patch_tokens.squeeze(0) # [64, 64, 768]
        dino_obs_feat = dino_obs_feat / dino_obs_feat.norm(dim=-1, keepdim=True)
        dino_obs_feat = dino_obs_feat.permute(2, 0, 1).reshape(768, -1)
        dino_sim = self.dino_target_feat @ dino_obs_feat
        dino_sim = dino_sim.reshape(1, 1, 64, 64)
        dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
        f_dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
            dino_sim,
            input_size=self.efficientvit_sam_predictor.input_size,
            original_size=self.efficientvit_sam_predictor.original_size,
        ).squeeze()

        self.efficientvit_sam_predictor.set_image(obs.transpose(1, 2, 0))
        obs_feat = self.efficientvit_sam_predictor.features.squeeze()
        C, h, w = obs_feat.shape
        obs_feat = obs_feat / obs_feat.norm(dim=0, keepdim=True)
        obs_feat = obs_feat.reshape(C, -1)
        sim = self.target_feat @ obs_feat
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode='bilinear', align_corners=False)
        f_sim = self.efficientvit_sam_predictor.model.postprocess_masks(
            sim,
            input_size=self.efficientvit_sam_predictor.input_size,
            original_size=self.efficientvit_sam_predictor.original_size,
        ).squeeze()

        f_sim = (f_dino_sim + f_sim) / 2
        topk_xy, topk_label = point_selection(f_sim, topk=1)
        n_xy, n_label = negative_point_selection(f_sim, topk=1)
        topk_xy = np.concatenate((topk_xy, n_xy), axis=0)
        topk_label = np.concatenate((topk_label, n_label), axis=0)

        for i in range(len(self.dino_parts_target_feat)):
            dino_target_feat = self.dino_parts_target_feat[i]
            dino_sim = dino_target_feat @ dino_obs_feat
            dino_sim = dino_sim.reshape(1, 1, 64, 64)
            dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
            dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
                dino_sim,
                input_size=self.efficientvit_sam_predictor.input_size,
                original_size=self.efficientvit_sam_predictor.original_size,
            ).squeeze()

            target_feat = self.part_target_feat[i]
            sim = target_feat @ obs_feat
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode='bilinear', align_corners=False)
            sim = self.efficientvit_sam_predictor.model.postprocess_masks(
                sim,
                input_size=self.efficientvit_sam_predictor.input_size,
                original_size=self.efficientvit_sam_predictor.original_size,
            ).squeeze()

            sim = (dino_sim + sim) / 2
            xy_, label_ = point_selection(sim, topk=1)
            topk_xy = np.concatenate((topk_xy, xy_), axis=0)
            topk_label = np.concatenate((topk_label, label_), axis=0)
            
        masks, scores, logits, logits_high = self.efficientvit_sam_predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()
        logits = logits * self.weights_np[..., None]
        logit = logits.sum(0)

        y, x = np.nonzero(mask)
        if len(y) == 0 or len(x) == 0:
            input_box = np.array([1, 1, 83, 83])
        else:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])

        n_point, n_label = negative_point_selection(f_sim, topk=1, box=input_box)
        topk_xy = np.concatenate((topk_xy, n_point), axis=0)
        topk_label = np.concatenate((topk_label, n_label), axis=0)

        masks, scores, logits, _ = self.efficientvit_sam_predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logit[None, :, :],
            multimask_output=True
        )
        best_idx = np.argmax(scores)
        y, x = np.nonzero(masks[best_idx])
        if len(y) == 0 or len(x) == 0:
            input_box = np.array([1, 1, 83, 83])
        else:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])

        n_point, n_label = negative_point_selection(f_sim, topk=1)
        topk_xy = np.concatenate((topk_xy, n_point), axis=0)
        topk_label = np.concatenate((topk_label, n_label), axis=0)

        masks, scores, logits, _ = self.efficientvit_sam_predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)

        mask = masks[best_idx]
        obs = obs * mask
        return torch.from_numpy(obs).to(self.device)
        
    def step(self, mdp_data: MdpData) -> MdpData:
        images = mdp_data.data[self.in_key].squeeze(1).clone()  # (b,c,h,w)
        mdp_data.data[self.out_key] = self._extract_pixels(images).unsqueeze(0).unsqueeze(0)
        return mdp_data
