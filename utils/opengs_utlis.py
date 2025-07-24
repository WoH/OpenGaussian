import torch
import numpy as np
import torch.nn.functional as F
import os
from bitarray import bitarray
from collections import OrderedDict

def calculate_pairwise_distances(tensor1, tensor2, metric=None):
    """
    Calculate L1 (Manhattan) and L2 (Euclidean) distances between every pair of vectors
    in two tensors of shape [m, 6] and [n, 6].
    Args:
        tensor1 (torch.Tensor): A tensor of shape [m, 6].
        tensor2 (torch.Tensor): Another tensor of shape [n, 6].
    Returns:
        torch.Tensor: L1 distances of shape [m, n].
        torch.Tensor: L2 distances of shape [m, n].
    """
    # Reshape tensors to allow broadcasting
    # tensor1 shape becomes [m, 1, 6] and tensor2 shape becomes [1, n, 6]
    tensor1 = tensor1.unsqueeze(1)  # Now tensor1 is [m, 1, 6]
    tensor2 = tensor2.unsqueeze(0)  # Now tensor2 is [1, n, 6]

    # Compute the L1 distance
    if metric == "l1":
        return torch.abs(tensor1 - tensor2).sum(dim=2), None  # Result is [m, n]

    # Compute the L2 distance
    if metric == "l2":
        return None, torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))  # Result is [m, n]

    l1_distances = torch.abs(tensor1 - tensor2).sum(dim=2)
    l2_distances = torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))
    return l1_distances, l2_distances

def calculate_distances(tensor1, tensor2, metric=None):
    """
    Calculate L1 (Manhattan) and L2 (Euclidean) distances between corresponding vectors
    in two tensors of shape [N, dim].
    Args:
        tensor1 (torch.Tensor): A tensor of shape [N, dim].
        tensor2 (torch.Tensor): Another tensor of shape [N, dim].
    Returns:
        torch.Tensor: L1 distances of shape [N].
        torch.Tensor: L2 distances of shape [N].
    """
    # Compute L1 distance
    if metric == "l1":
        return torch.abs(tensor1 - tensor2).sum(dim=1)
    
    # Compute L2 distance
    if metric == "l2":
        return torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=1))
    
    l1_distances = torch.abs(tensor1 - tensor2).sum(dim=1)
    l2_distances = torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=1))

    return l1_distances, l2_distances
    

def bin2dec(b, bits):
    """Convert binary b to decimal integer.
    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, torch.int64)
    return torch.sum(mask * b, -1)

def load_code_book(base_path):
    inds_file = os.path.join(base_path, 'kmeans_inds.bin')
    codebook_file = os.path.join(base_path, 'kmeans_centers.pth')
    args_file = os.path.join(base_path, 'kmeans_args.npy')
    codebook = torch.load(codebook_file)    # [num_cluster, dim]
    args_dict = np.load(args_file, allow_pickle=True).item()
    quant_params = args_dict['params']
    loaded_bitarray = bitarray()
    with open(inds_file, 'rb') as file:
        loaded_bitarray.fromfile(file)
    # bitarray pads 0s if array is not divisible by 8. ignore extra 0s at end when loading
    total_len = args_dict['total_len']
    loaded_bitarray = loaded_bitarray[:total_len].tolist()
    indices = np.reshape(loaded_bitarray, (-1, args_dict['n_bits']))
    indices = bin2dec(torch.from_numpy(indices), args_dict['n_bits'])
    indices = np.reshape(indices.cpu().numpy(), (len(quant_params), -1))
    indices_dict = OrderedDict()
    for i, key in enumerate(args_dict['params']):
        indices_dict[key] = indices[i]
    
    return codebook, indices_dict['ins_feat']

def calculate_iou(masks1, masks2, base=None):
    """
    Calculate the Intersection over Union (IoU) between two sets of masks.
    Args:
        masks1: PyTorch tensor of shape [n, H, W], torch.int32.
        masks2: PyTorch tensor of shape [m, H, W], torch.int32.
    Returns:
        iou_matrix: PyTorch tensor of shape [m, n], containing IoU values.
    """
    # Ensure the masks are of type torch.int32
    if masks1.dtype != torch.bool:
        masks1 = masks1.to(torch.bool)
    if masks2.dtype != torch.bool:
        masks2 = masks2.to(torch.bool)
    
    # Expand masks to broadcastable shapes
    masks1_expanded = masks1.unsqueeze(0)  # [1, n, H, W]
    masks2_expanded = masks2.unsqueeze(1)  # [m, 1, H, W]
    
    # Compute intersection
    intersection = (masks1_expanded & masks2_expanded).float().sum(dim=(2, 3))  # [m, n]
    
    # Compute union
    if base == "former":
        union = (masks1_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    elif base == "later":
        union = (masks2_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    else:
        union = (masks1_expanded | masks2_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    
    # Compute IoU
    iou_matrix = intersection / union
    
    return iou_matrix

def get_SAM_mask_and_feat(gt_sam_mask, level=3, filter_th=50, original_mask_feat=None, sample_mask=False):
    """
    input: 
        gt_sam_mask[4, H, W]: mask id
    output:
        mask_id[H, W]: The ID of the mask each pixel belongs to (0 indicates invalid pixels)
        mask_bool[num_mask+1, H, W]: Boolean, note that the return value excludes the 0th mask (invalid points)
        invalid_pix[H, W]: Boolean, invalid pixels
    """
    # (1) mask id: -1, 1, 2, 3,...
    mask_id = gt_sam_mask[level].clone()
    if level > 0:
        # subtract the maximum mask ID of the previous level
        mask_id = mask_id - (gt_sam_mask[level-1].max().detach().cpu()+1)
    if mask_id.min() < 0:
        mask_id = mask_id.clamp_min(-1)    # -1, 0~num_mask
    mask_id += 1    # 0, 1~num_mask+1
    invalid_pix = mask_id==0    # invalid pixels

    # (2) mask id[H, W] -> one-hot/mask_bool [num_mask+1, H, W]
    instance_num = mask_id.max()
    one_hot = F.one_hot(mask_id.type(torch.int64), num_classes=int(instance_num.item() + 1))
    # bool mask [num+1, H, W]
    mask_bool = one_hot.permute(2, 0, 1)
    
    # # TODO modify -------- only keep the largest 50
    # if instance_num > 50:
    #     top50_values, _ = torch.topk(mask_bool.sum(dim=(1,2)), 50, largest=True)
    #     filter_th = top50_values[-1].item()
    # # modify --------

    # # TODO: not used
    # # (3) delete small mask 
    # saved_idx = mask_bool.sum(dim=(1,2)) >= filter_th  # default 50 pixels
    # # Random sampling, not actually used
    # if sample_mask:
    #     prob = torch.rand(saved_idx.shape[0])
    #     sample_ind = prob > 0.5
    #     saved_idx = saved_idx & sample_ind.cuda()
    # saved_idx[0] = True  # Keep the mask for invalid points, ensuring that mask_id == 0 corresponds to invalid pixels.
    # mask_bool = mask_bool[saved_idx]    # [num_filt, H, W]

    # update mask id
    mask_id = torch.argmax(mask_bool, dim=0)  # [H, W] The ID of the pixels after filtering is 0
    invalid_pix = mask_id==0

    # TODO not used!
    # (4) Get the language features corresponding to the masks (used for 2D-3D association in the third stage)
    if original_mask_feat is not None:
        mask_feat = original_mask_feat.clone()       # [num_mask, 512]
        max_ind = int(gt_sam_mask[level].max())+1
        min_ind = int(gt_sam_mask[level-1].max())+1 if level > 0 else 0
        mask_feat = mask_feat[min_ind:max_ind, :]
        # # update mask feat
        # mask_feat = mask_feat[saved_idx[1:]]    # The 0th element of saved_idx is the mask corresponding to invalid pixels and has no features

        return mask_id, mask_bool[1:, :, :], mask_feat, invalid_pix
    return mask_id, mask_bool[1:, :, :], invalid_pix

def pair_mask_feature_mean(feat_map, masks):
    """ mean feat of N masks
    feat_map: [N, C, H, W]
    masks: [N, H, W]
    mean_values: [N, C]
    """
    N, C, H, W = feat_map.shape

    # [N, H, W] -> [N, C, H, W]
    expanded_masks = masks.unsqueeze(1).expand(-1, C, -1, -1)
    # [N, C, H, W]
    masked_features = feat_map * expanded_masks.float()
    # pixels
    mask_counts = expanded_masks.sum(dim=[2, 3]) + 1e-6
    # mean feat [N, C]
    mean_values = masked_features.sum(dim=[2, 3]) / mask_counts

    return mean_values

def process_in_chunks(masks_expanded, masked_feats, mean_per_channel, chunk_size=5):
    result = torch.zeros_like(masked_feats)
    for i in range(0, masks_expanded.size(0), chunk_size):
        end_i = min(i + chunk_size, masks_expanded.size(0))
        for j in range(0, masks_expanded.size(1), chunk_size):
            end_j = min(j + chunk_size, masks_expanded.size(1))
            chunk_mask = masks_expanded[i:end_i, j:end_j]
            chunk_feats = masked_feats[i:end_i, j:end_j]
            chunk_mean = mean_per_channel[i:end_i, j:end_j].unsqueeze(-1).unsqueeze(-1)

            result[i:end_i, j:end_j] = torch.where(chunk_mask.bool(), chunk_feats - chunk_mean, torch.zeros_like(chunk_feats))
    return result

def calculate_variance_in_chunks(masked_for_variance, mask_counts, chunk_size=5):
    variance_per_channel = torch.zeros(masked_for_variance.size(0), masked_for_variance.size(1), device=masked_for_variance.device)
    for i in range(0, masked_for_variance.size(0), chunk_size):
        end_i = min(i + chunk_size, masked_for_variance.size(0))
        for j in range(0, masked_for_variance.size(1), chunk_size):
            end_j = min(j + chunk_size, masked_for_variance.size(1))
            chunk_masked_for_variance = masked_for_variance[i:end_i, j:end_j]

            chunk_variance = (chunk_masked_for_variance ** 2).sum(dim=[2, 3]) / mask_counts[i:end_i, j:end_j]
            variance_per_channel[i:end_i, j:end_j] = chunk_variance
    return variance_per_channel

def ele_multip_in_chunks(feat_expanded, masks_expanded, chunk_size=5):
    result = torch.zeros_like(feat_expanded)
    for i in range(0, feat_expanded.size(0), chunk_size):
        end_i = min(i + chunk_size, feat_expanded.size(0))
        for j in range(0, feat_expanded.size(1), chunk_size):
            end_j = min(j + chunk_size, feat_expanded.size(1))
            chunk_feat = feat_expanded[i:end_i, j:end_j]
            chunk_mask = masks_expanded[i:end_i, j:end_j].float()

            result[i:end_i, j:end_j] = chunk_feat * chunk_mask
    return result

def mask_feature_mean(feat_map, gt_masks, image_mask=None, return_var=False):
    """Compute the average instance features within each mask.
    feat_map: [C=6, H, W]         the instance features of the entire image
    gt_masks: [num_mask, H, W]  num_mask boolean masks
    """
    num_mask, H, W = gt_masks.shape

    # expand feat and masks for batch processing
    feat_expanded = feat_map.unsqueeze(0).expand(num_mask, *feat_map.shape)  # [num_mask, C, H, W]
    masks_expanded = gt_masks.unsqueeze(1).expand(-1, feat_map.shape[0], -1, -1)  # [num_mask, C, H, W]
    if image_mask is not None:  # image level mask
        image_mask_expanded = image_mask.unsqueeze(0).expand(num_mask, feat_map.shape[0], -1, -1)

    # average features within each mask
    if image_mask is not None:
        masked_feats = feat_expanded * masks_expanded.float() * image_mask_expanded.float()
        mask_counts = (masks_expanded * image_mask_expanded.float()).sum(dim=(2, 3))
    else:
        # masked_feats = feat_expanded * masks_expanded.float()  # [num_mask, C, H, W] may cause OOM
        masked_feats = ele_multip_in_chunks(feat_expanded, masks_expanded, chunk_size=5)   # in chuck to avoid OOM
        mask_counts = masks_expanded.sum(dim=(2, 3))  # [num_mask, C]

    # the number of pixels within each mask
    mask_counts = mask_counts.clamp(min=1)

    # the mean features of each mask
    sum_per_channel = masked_feats.sum(dim=[2, 3])
    mean_per_channel = sum_per_channel / mask_counts    # [num_mask, C]

    if not return_var:
        return mean_per_channel   # [num_mask, C]
    else:
        # calculate variance
        # masked_for_variance = torch.where(masks_expanded.bool(), masked_feats - mean_per_channel.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(masked_feats))
        masked_for_variance = process_in_chunks(masks_expanded, masked_feats, mean_per_channel, chunk_size=5) # in chunk to avoid OOM

        # variance_per_channel = (masked_for_variance ** 2).sum(dim=[2, 3]) / mask_counts    # [num_mask, 6]
        variance_per_channel = calculate_variance_in_chunks(masked_for_variance, mask_counts, chunk_size=5)   # in chuck to avoid OOM

        # mean and variance
        mean = mean_per_channel.mean(dim=1)          # [num_mask]，not used
        variance = variance_per_channel.mean(dim=1)  # [num_mask]

        return mean_per_channel, variance, mask_counts[:, 0]   # [num_mask, C], [num_mask], [num_mask]

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb):
    if isinstance(srgb, torch.Tensor):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = torch.clamp(((200 * srgb + 11) / (211)), min=eps)**(12 / 5)
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = np.finfo(np.float32).eps
        linear0 = 25 / 323 * srgb
        linear1 = np.maximum(((200 * srgb + 11) / (211)), eps)**(12 / 5)
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError


# Multi-View CLIP Feature Fusion Functions

def compute_gaussian_visibility(gaussians_xyz, cameras, depth_maps, gaussian_scales=None):
    """
    Compute which views each 3D Gaussian is visible from.
    
    Args:
        gaussians_xyz: [N_gaussians, 3] - 3D positions of Gaussians
        cameras: List of camera objects
        depth_maps: Dict {view_idx: depth[H, W]} - depth maps for each view
        gaussian_scales: [N_gaussians, 3] - scales of Gaussians (optional, for better occlusion testing)
    
    Returns:
        visibility_matrix: [N_gaussians, N_views] - boolean visibility matrix
        projected_coords: Dict {view_idx: [N_gaussians, 2]} - 2D projections per view
    """
    N_gaussians = gaussians_xyz.shape[0]
    N_views = len(cameras)
    visibility_matrix = torch.zeros(N_gaussians, N_views, dtype=torch.bool, device=gaussians_xyz.device)
    projected_coords = {}
    
    for view_idx, camera in enumerate(cameras):
        # Project 3D points to 2D
        gaussians_xyz_hom = torch.cat([gaussians_xyz, torch.ones(N_gaussians, 1, device=gaussians_xyz.device)], dim=1)
        
        # Transform to camera space
        cam_coords = (camera.world_view_transform @ gaussians_xyz_hom.T).T  # [N_gaussians, 4]
        
        # Project to screen space
        screen_coords = (camera.projection_matrix @ cam_coords.T).T  # [N_gaussians, 4]
        
        # Perspective divide (avoid division by zero)
        w_coords = screen_coords[:, 3:4]
        w_coords = torch.where(torch.abs(w_coords) < 1e-6, torch.sign(w_coords) * 1e-6, w_coords)
        screen_coords = screen_coords / w_coords  # [N_gaussians, 4]
        
        # Convert to pixel coordinates
        H, W = depth_maps[view_idx].shape if view_idx in depth_maps else (480, 640)  # Default resolution
        pixel_coords = torch.zeros(N_gaussians, 2, device=gaussians_xyz.device)
        pixel_coords[:, 0] = (screen_coords[:, 0] + 1.0) * 0.5 * W
        pixel_coords[:, 1] = (screen_coords[:, 1] + 1.0) * 0.5 * H
        
        projected_coords[view_idx] = pixel_coords.clone()
        
        # Check bounds
        in_bounds = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) & \
                   (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
        
        # Check depth (if depth map available)
        if view_idx in depth_maps:
            depth_map = depth_maps[view_idx]
            pixel_coords_int = pixel_coords.long().clamp(min=0)
            pixel_coords_int[:, 0] = pixel_coords_int[:, 0].clamp(max=W-1)
            pixel_coords_int[:, 1] = pixel_coords_int[:, 1].clamp(max=H-1)
            
            # Get depth at projected locations
            projected_depths = depth_map[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
            gaussian_depths = cam_coords[:, 2]  # Z coordinate in camera space
            
            # Account for Gaussian scale in depth testing (optional)
            depth_tolerance = 0.1  # Default tolerance
            if gaussian_scales is not None:
                # Use maximum scale as depth tolerance
                depth_tolerance = gaussian_scales.max(dim=1)[0] * 0.5
            
            depth_visible = torch.abs(gaussian_depths - projected_depths) < depth_tolerance
        else:
            # No depth testing, assume visible if in bounds and in front of camera
            depth_visible = cam_coords[:, 2] > 0.1  # Must be in front of camera
        
        # Combine bounds and depth checks
        visibility_matrix[:, view_idx] = in_bounds & depth_visible
    
    return visibility_matrix, projected_coords


def aggregate_clip_features_per_view(sam_masks_dict, clip_features_dict, projected_coords, visibility_matrix):
    """
    Assign CLIP features to Gaussians based on SAM mask membership.
    
    Args:
        sam_masks_dict: Dict {view_idx: mask_bool[num_masks, H, W]} - SAM masks per view
        clip_features_dict: Dict {view_idx: clip_features[num_masks, 512]} - CLIP features per mask per view
        projected_coords: Dict {view_idx: [N_gaussians, 2]} - 2D projections of Gaussians
        visibility_matrix: [N_gaussians, N_views] - boolean visibility matrix
    
    Returns:
        gaussian_clip_features: Dict {view_idx: [N_gaussians, 512]} - CLIP features per Gaussian per view
    """
    N_gaussians = visibility_matrix.shape[0]
    gaussian_clip_features = {}
    
    for view_idx in sam_masks_dict.keys():
        if view_idx not in clip_features_dict or view_idx not in projected_coords:
            continue
            
        sam_masks = sam_masks_dict[view_idx]  # [num_masks, H, W]
        clip_features = clip_features_dict[view_idx]  # [num_masks, 512]
        coords = projected_coords[view_idx]  # [N_gaussians, 2]
        
        # Initialize with zero features
        view_features = torch.zeros(N_gaussians, 512, device=clip_features.device)
        
        # Only process visible Gaussians
        visible_mask = visibility_matrix[:, view_idx]
        visible_coords = coords[visible_mask]
        
        if visible_mask.sum() == 0:
            gaussian_clip_features[view_idx] = view_features
            continue
        
        # Convert to integer coordinates
        H, W = sam_masks.shape[1], sam_masks.shape[2]
        pixel_coords = visible_coords.long().clamp(min=0)
        pixel_coords[:, 0] = pixel_coords[:, 0].clamp(max=W-1)
        pixel_coords[:, 1] = pixel_coords[:, 1].clamp(max=H-1)
        
        # Find which mask each visible Gaussian belongs to
        mask_assignments = torch.zeros(visible_mask.sum(), dtype=torch.long, device=sam_masks.device)
        
        for i, (y, x) in enumerate(pixel_coords):
            # Check all masks at this pixel location
            mask_values = sam_masks[:, y, x]  # [num_masks]
            
            # Find the first mask that contains this pixel (masks are mutually exclusive)
            mask_indices = torch.where(mask_values)[0]
            if len(mask_indices) > 0:
                mask_assignments[i] = mask_indices[0]
            # else: remains 0 (background/no mask)
        
        # Assign CLIP features based on mask assignments
        visible_features = torch.zeros(visible_mask.sum(), 512, device=clip_features.device)
        
        for mask_idx in range(clip_features.shape[0]):
            mask_gaussians = (mask_assignments == mask_idx)
            if mask_gaussians.sum() > 0:
                visible_features[mask_gaussians] = clip_features[mask_idx]
        
        # Place visible features back into full feature tensor
        view_features[visible_mask] = visible_features
        gaussian_clip_features[view_idx] = view_features
    
    return gaussian_clip_features


def compute_view_weights(gaussians_xyz, cameras, visibility_matrix, projected_coords):
    """
    Compute reliability weights for each view based on distance and viewing angle.
    
    Args:
        gaussians_xyz: [N_gaussians, 3] - 3D positions of Gaussians
        cameras: List of camera objects
        visibility_matrix: [N_gaussians, N_views] - boolean visibility matrix
        projected_coords: Dict {view_idx: [N_gaussians, 2]} - 2D projections per view
        
    Returns:
        weights: [N_gaussians, N_views] - normalized weights for visible pairs
    """
    N_gaussians, N_views = visibility_matrix.shape
    weights = torch.zeros_like(visibility_matrix, dtype=torch.float32)
    
    for view_idx, camera in enumerate(cameras):
        if view_idx not in projected_coords:
            continue
        
        visible_mask = visibility_matrix[:, view_idx]
        if not visible_mask.any():
            continue
        
        visible_gaussians = gaussians_xyz[visible_mask]
        
        # Compute camera center in world coordinates
        camera_center = camera.camera_center  # [3]
        
        # Distance weighting (closer = higher weight)
        distances = torch.norm(visible_gaussians - camera_center, dim=1)  # [N_visible]
        distance_weights = 1.0 / (distances + 1e-6)  # Inverse distance
        
        # Viewing angle weighting (front-facing = higher weight)
        # Compute view directions
        view_directions = F.normalize(visible_gaussians - camera_center, dim=1)  # [N_visible, 3]
        
        # Camera forward direction (assuming camera looks down negative Z)
        camera_forward = camera.world_view_transform[:3, 2]  # Camera's Z axis in world coords
        camera_forward = -F.normalize(camera_forward.unsqueeze(0), dim=1)  # [1, 3]
        
        # Compute dot product (cosine of angle)
        angle_weights = torch.clamp(torch.sum(view_directions * camera_forward, dim=1), min=0.0)  # [N_visible]
        
        # Combine weights
        combined_weights = distance_weights * angle_weights
        
        # Place back into full weight tensor
        weights[visible_mask, view_idx] = combined_weights
    
    # Normalize weights per Gaussian (so they sum to 1 across views)
    row_sums = weights.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    weights = weights / row_sums
    
    return weights


def fuse_multiview_clip_features(gaussian_clip_features, visibility_matrix, view_weights):
    """
    Average CLIP features across visible views with reliability weighting.
    
    Args:
        gaussian_clip_features: Dict {view_idx: [N_gaussians, 512]} - CLIP features per view
        visibility_matrix: [N_gaussians, N_views] - boolean visibility matrix
        view_weights: [N_gaussians, N_views] - normalized reliability weights
        
    Returns:
        fused_features: [N_gaussians, 512] - fused CLIP features
    """
    if not gaussian_clip_features:
        return torch.zeros(visibility_matrix.shape[0], 512)
    
    N_gaussians = visibility_matrix.shape[0]
    feature_dim = 512
    device = next(iter(gaussian_clip_features.values())).device
    
    fused_features = torch.zeros(N_gaussians, feature_dim, device=device)
    
    # Stack all view features and weights
    view_indices = sorted(gaussian_clip_features.keys())
    stacked_features = torch.stack([gaussian_clip_features[idx] for idx in view_indices], dim=2)  # [N_gaussians, 512, N_valid_views]
    stacked_weights = view_weights[:, view_indices]  # [N_gaussians, N_valid_views]
    stacked_visibility = visibility_matrix[:, view_indices]  # [N_gaussians, N_valid_views]
    
    # Apply visibility mask to weights
    masked_weights = stacked_weights * stacked_visibility.float()
    
    # Weighted average across views
    weighted_features = stacked_features * masked_weights.unsqueeze(1)  # [N_gaussians, 512, N_valid_views]
    fused_features = weighted_features.sum(dim=2)  # [N_gaussians, 512]
    
    # Handle Gaussians that are not visible in any view
    visible_anywhere = visibility_matrix.any(dim=1)
    invisible_gaussians = ~visible_anywhere
    
    if invisible_gaussians.any():
        # For invisible Gaussians, use zero features or learned embeddings
        # This could be enhanced with learned "default" semantic embeddings
        fused_features[invisible_gaussians] = 0.0
    
    return fused_features