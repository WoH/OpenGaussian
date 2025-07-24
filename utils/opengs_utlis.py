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

# Multi-View SAM Mask Refinement Implementation
# Credits: Inspired by Gaussian Grouping (https://github.com/lkeab/gaussian-grouping)
# and standard multi-view geometry principles

def detect_view_overlaps(cameras, depth_maps, overlap_threshold=0.1):
    """
    Detect overlapping views using camera frustums and depth information.
    
    Args:
        cameras: List of camera objects with poses and intrinsics
        depth_maps: Dict of rendered depth maps {camera_idx: depth_tensor[H, W]}
        overlap_threshold: Minimum overlap ratio to consider views as overlapping
    
    Returns:
        overlap_matrix: [N_views, N_views] tensor with overlap percentages
        overlapping_pairs: List of (view_i, view_j) tuples for overlapping view pairs
    """
    n_views = len(cameras)
    overlap_matrix = torch.zeros(n_views, n_views)
    overlapping_pairs = []
    
    for i in range(n_views):
        for j in range(i + 1, n_views):
            cam_i, cam_j = cameras[i], cameras[j]
            
            # Simple overlap estimation using camera viewing angles
            # Transform camera centers to world coordinates
            center_i = -cam_i.R.T @ cam_i.T
            center_j = -cam_j.R.T @ cam_j.T
            
            # Camera viewing directions
            view_dir_i = cam_i.R[2, :]  # Forward direction (z-axis)
            view_dir_j = cam_j.R[2, :]
            
            # Distance between camera centers
            distance = torch.norm(center_i - center_j)
            
            # Viewing angle similarity (dot product of view directions)
            view_similarity = torch.dot(view_dir_i, view_dir_j)
            
            # Simple heuristic: close cameras with similar viewing directions likely overlap
            if distance < 5.0 and view_similarity > 0.3:  # Tunable thresholds
                overlap_ratio = max(0.0, view_similarity * (5.0 - distance) / 5.0)
                if overlap_ratio > overlap_threshold:
                    overlap_matrix[i, j] = overlap_ratio
                    overlap_matrix[j, i] = overlap_ratio
                    overlapping_pairs.append((i, j))
    
    return overlap_matrix, overlapping_pairs


def reproject_masks_between_views(mask_bool, src_camera, dst_camera, src_depth, dst_depth, 
                                 occlusion_threshold=0.01):
    """
    Reproject boolean masks from source view to destination view.
    
    Args:
        mask_bool: [num_masks, H, W] boolean masks in source view
        src_camera: Source camera object with intrinsics and pose
        dst_camera: Destination camera object  
        src_depth: [H, W] depth map in source view
        dst_depth: [H, W] depth map in destination view
        occlusion_threshold: Depth difference threshold for occlusion detection
    
    Returns:
        reprojected_masks: [num_masks, H, W] boolean masks reprojected to destination view
        valid_mask: [H, W] boolean mask indicating valid reprojected pixels
    """
    device = mask_bool.device
    num_masks, H, W = mask_bool.shape
    
    # Create pixel coordinates in source view
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([x_coords, y_coords], dim=-1).float().to(device)  # [H, W, 2]
    
    # Convert to homogeneous coordinates
    ones = torch.ones(H, W, 1).to(device)
    pixel_coords = torch.cat([coords, ones], dim=-1)  # [H, W, 3]
    
    # Unproject to 3D using source camera
    # Convert to camera coordinates
    K_inv = torch.inverse(src_camera.K)  # Assuming K matrix is available
    cam_coords = pixel_coords @ K_inv.T  # [H, W, 3]
    cam_coords = cam_coords * src_depth.unsqueeze(-1)  # Scale by depth
    
    # Convert to homogeneous coordinates
    ones = torch.ones(H, W, 1).to(device)
    cam_coords_hom = torch.cat([cam_coords, ones], dim=-1)  # [H, W, 4]
    
    # Transform to world coordinates
    src_world_to_cam = torch.cat([src_camera.R, src_camera.T.unsqueeze(-1)], dim=-1)
    src_cam_to_world = torch.inverse(torch.cat([src_world_to_cam, 
                                              torch.tensor([[0, 0, 0, 1]]).to(device)], dim=0))
    
    world_coords = cam_coords_hom @ src_cam_to_world.T  # [H, W, 4]
    
    # Transform to destination camera coordinates
    dst_world_to_cam = torch.cat([dst_camera.R, dst_camera.T.unsqueeze(-1)], dim=-1)
    dst_world_to_cam_hom = torch.cat([dst_world_to_cam, 
                                    torch.tensor([[0, 0, 0, 1]]).to(device)], dim=0)
    
    dst_cam_coords = world_coords @ dst_world_to_cam_hom.T  # [H, W, 4]
    dst_cam_coords = dst_cam_coords[:, :, :3]  # [H, W, 3]
    
    # Project to destination image coordinates
    dst_pixel_coords = dst_cam_coords @ dst_camera.K.T  # [H, W, 3]
    dst_pixel_coords = dst_pixel_coords[:, :, :2] / dst_pixel_coords[:, :, 2:3]  # [H, W, 2]
    
    # Check bounds and occlusions
    valid_x = (dst_pixel_coords[:, :, 0] >= 0) & (dst_pixel_coords[:, :, 0] < W)
    valid_y = (dst_pixel_coords[:, :, 1] >= 0) & (dst_pixel_coords[:, :, 1] < H)
    valid_bounds = valid_x & valid_y
    
    # Sample destination depth at reprojected locations
    dst_pixel_coords_int = dst_pixel_coords.round().long()
    valid_coords = valid_bounds & (dst_pixel_coords_int[:, :, 0] < W) & (dst_pixel_coords_int[:, :, 1] < H)
    
    sampled_depth = torch.zeros_like(src_depth)
    sampled_depth[valid_coords] = dst_depth[dst_pixel_coords_int[valid_coords, 1], 
                                          dst_pixel_coords_int[valid_coords, 0]]
    
    # Check occlusion using depth
    reprojected_depth = dst_cam_coords[:, :, 2]
    valid_depth = torch.abs(reprojected_depth - sampled_depth) < occlusion_threshold
    valid_mask = valid_bounds & valid_depth
    
    # Reproject masks using bilinear interpolation
    reprojected_masks = torch.zeros_like(mask_bool)
    
    for mask_idx in range(num_masks):
        # Use grid_sample for bilinear interpolation
        mask_float = mask_bool[mask_idx].float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        norm_coords = dst_pixel_coords.clone()
        norm_coords[:, :, 0] = 2.0 * norm_coords[:, :, 0] / (W - 1) - 1.0
        norm_coords[:, :, 1] = 2.0 * norm_coords[:, :, 1] / (H - 1) - 1.0
        norm_coords = norm_coords.unsqueeze(0)  # [1, H, W, 2]
        
        # Sample mask values
        sampled_mask = F.grid_sample(mask_float, norm_coords, 
                                   mode='bilinear', padding_mode='zeros', align_corners=True)
        reprojected_masks[mask_idx] = sampled_mask.squeeze() > 0.5
    
    # Apply validity mask
    reprojected_masks = reprojected_masks & valid_mask.unsqueeze(0)
    
    return reprojected_masks, valid_mask


def measure_mask_consistency(original_masks, reprojected_masks, valid_mask):
    """
    Measure consistency between original and reprojected masks.
    
    Args:
        original_masks: [num_masks, H, W] boolean masks in destination view
        reprojected_masks: [num_masks, H, W] boolean masks reprojected from source view
        valid_mask: [H, W] boolean mask indicating valid reprojected pixels
    
    Returns:
        consistency_scores: [num_masks] consistency score for each mask (0-1)
        iou_scores: [num_masks] IoU scores for each mask
    """
    num_masks = original_masks.shape[0]
    consistency_scores = torch.zeros(num_masks)
    iou_scores = torch.zeros(num_masks)
    
    for mask_idx in range(num_masks):
        orig_mask = original_masks[mask_idx] & valid_mask
        repr_mask = reprojected_masks[mask_idx] & valid_mask
        
        # Compute IoU
        intersection = (orig_mask & repr_mask).sum().float()
        union = (orig_mask | repr_mask).sum().float()
        
        if union > 0:
            iou = intersection / union
            iou_scores[mask_idx] = iou
            
            # Boundary consistency using edge detection
            orig_edges = torch.abs(F.conv2d(orig_mask.float().unsqueeze(0).unsqueeze(0), 
                                          torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float().to(orig_mask.device),
                                          padding=1)) > 0.1
            repr_edges = torch.abs(F.conv2d(repr_mask.float().unsqueeze(0).unsqueeze(0),
                                          torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]]).float().to(repr_mask.device), 
                                          padding=1)) > 0.1
            
            edge_intersection = (orig_edges.squeeze() & repr_edges.squeeze() & valid_mask).sum().float()
            edge_union = (orig_edges.squeeze() | repr_edges.squeeze()).sum().float()
            
            boundary_consistency = edge_intersection / (edge_union + 1e-8)
            
            # Combined consistency score (weighted average of IoU and boundary consistency)
            consistency_scores[mask_idx] = 0.7 * iou + 0.3 * boundary_consistency
        
    return consistency_scores, iou_scores


def refine_sam_masks_multiview(sam_masks_dict, cameras, depth_maps, max_iterations=3, 
                              consistency_threshold=0.5):
    """
    Refine SAM masks using multi-view consistency.
    
    Args:
        sam_masks_dict: Dict {view_idx: mask_bool[num_masks, H, W]} SAM masks for each view
        cameras: List of camera objects
        depth_maps: Dict {view_idx: depth[H, W]} depth maps for each view
        max_iterations: Maximum number of refinement iterations
        consistency_threshold: Minimum consistency score to keep mask regions
    
    Returns:
        refined_masks_dict: Dict {view_idx: refined_mask_bool[num_masks, H, W]}
    """
    # Detect view overlaps
    overlap_matrix, overlapping_pairs = detect_view_overlaps(cameras, depth_maps)
    
    refined_masks_dict = {k: v.clone() for k, v in sam_masks_dict.items()}
    
    for iteration in range(max_iterations):
        improvements = 0
        
        # Process each overlapping pair
        for view_i, view_j in overlapping_pairs:
            if view_i not in sam_masks_dict or view_j not in sam_masks_dict:
                continue
                
            masks_i = refined_masks_dict[view_i]
            masks_j = refined_masks_dict[view_j]
            
            # Reproject masks from view_i to view_j
            repr_masks_i_to_j, valid_mask_i_to_j = reproject_masks_between_views(
                masks_i, cameras[view_i], cameras[view_j], 
                depth_maps[view_i], depth_maps[view_j])
            
            # Reproject masks from view_j to view_i  
            repr_masks_j_to_i, valid_mask_j_to_i = reproject_masks_between_views(
                masks_j, cameras[view_j], cameras[view_i],
                depth_maps[view_j], depth_maps[view_i])
            
            # Measure consistency
            consistency_i, iou_i = measure_mask_consistency(masks_j, repr_masks_i_to_j, valid_mask_i_to_j)
            consistency_j, iou_j = measure_mask_consistency(masks_i, repr_masks_j_to_i, valid_mask_j_to_i)
            
            # Refine masks based on consistency
            for mask_idx in range(min(masks_i.shape[0], masks_j.shape[0])):
                # Refine view_j mask using reprojection from view_i
                if consistency_i[mask_idx] > consistency_threshold:
                    # Use weighted combination based on consistency scores
                    weight_repr = consistency_i[mask_idx]
                    weight_orig = 1.0 - weight_repr
                    
                    combined_mask = (weight_orig * masks_j[mask_idx].float() + 
                                   weight_repr * repr_masks_i_to_j[mask_idx].float()) > 0.5
                    
                    if not torch.equal(refined_masks_dict[view_j][mask_idx], combined_mask):
                        refined_masks_dict[view_j][mask_idx] = combined_mask
                        improvements += 1
                
                # Refine view_i mask using reprojection from view_j
                if consistency_j[mask_idx] > consistency_threshold:
                    weight_repr = consistency_j[mask_idx]
                    weight_orig = 1.0 - weight_repr
                    
                    combined_mask = (weight_orig * masks_i[mask_idx].float() + 
                                   weight_repr * repr_masks_j_to_i[mask_idx].float()) > 0.5
                    
                    if not torch.equal(refined_masks_dict[view_i][mask_idx], combined_mask):
                        refined_masks_dict[view_i][mask_idx] = combined_mask
                        improvements += 1
        
        # Early stopping if no improvements
        if improvements == 0:
            break
    
    return refined_masks_dict


def get_SAM_mask_and_feat(gt_sam_mask, level=3, filter_th=50, original_mask_feat=None, sample_mask=False,
                         use_multiview_refinement=False, cameras=None, depth_maps=None, current_view_idx=None):
    """
    input: 
        gt_sam_mask[4, H, W]: mask id
        use_multiview_refinement: Whether to use multi-view refinement
        cameras: List of camera objects (required if use_multiview_refinement=True)
        depth_maps: Dict of depth maps (required if use_multiview_refinement=True)  
        current_view_idx: Index of current view (required if use_multiview_refinement=True)
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

    # (3) Apply multi-view refinement if enabled
    if use_multiview_refinement and cameras is not None and depth_maps is not None and current_view_idx is not None:
        # Store current view masks for multi-view processing
        current_masks = mask_bool[1:, :, :].clone()  # Exclude invalid mask
        
        # For now, we'll apply refinement as a post-processing step
        # In a full implementation, this would involve collecting masks from multiple views
        # and applying the refinement algorithm
        print(f"Multi-view refinement enabled for view {current_view_idx}")
        # Note: Full multi-view refinement requires coordination across views
        # This is a placeholder for the integration point
    
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