"""
Multi-view CLIP feature fusion module for OpenGaussian.
Aggregates CLIP features from multiple views with optional visibility weighting.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from scene.cameras import Camera
from utils.visibility import compute_visibility_weights


def load_clip_features(feature_path: str) -> Optional[torch.Tensor]:
    """
    Load CLIP features from .npy file.
    
    Args:
        feature_path: Path to the feature .npy file
        
    Returns:
        features: Tensor of shape (H, W, D) where D is feature dimension
    """
    if os.path.exists(feature_path):
        features = np.load(feature_path)
        return torch.from_numpy(features).float()
    else:
        return None


def get_gaussian_clip_features(gaussian_id: int,
                              cameras: List[Camera],
                              visible_indices: List[int],
                              projected_positions: List[Tuple[float, float]],
                              feature_dir: str) -> List[torch.Tensor]:
    """
    Extract CLIP features for a Gaussian from all visible views.
    
    Args:
        gaussian_id: ID of the Gaussian
        cameras: List of all Camera objects
        visible_indices: List of camera indices where Gaussian is visible
        projected_positions: List of (x, y) normalized coordinates for each visible view
        feature_dir: Directory containing CLIP feature files
        
    Returns:
        features: List of feature vectors for the Gaussian across views
    """
    features = []
    
    for idx, cam_idx in enumerate(visible_indices):
        camera = cameras[cam_idx]
        
        # Load CLIP features for this view
        feature_path = os.path.join(feature_dir, f"{camera.image_name}_f.npy")
        clip_features = load_clip_features(feature_path)
        
        if clip_features is None:
            print(f"Warning: CLIP features not found for {camera.image_name}")
            continue
            
        # Get projected position
        norm_x, norm_y = projected_positions[idx]
        
        # Convert to pixel coordinates
        pixel_x = int(norm_x * clip_features.shape[1])
        pixel_y = int(norm_y * clip_features.shape[0])
        
        # Clamp to valid range
        pixel_x = max(0, min(clip_features.shape[1] - 1, pixel_x))
        pixel_y = max(0, min(clip_features.shape[0] - 1, pixel_y))
        
        # Extract feature at this position
        feature = clip_features[pixel_y, pixel_x]
        features.append(feature)
    
    return features


def aggregate_multiview_features(features: List[torch.Tensor],
                                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Aggregate features from multiple views using weighted average.
    
    Args:
        features: List of feature tensors
        weights: Optional weights for each view (normalized to sum to 1)
        
    Returns:
        aggregated: Aggregated feature vector
    """
    if not features:
        raise ValueError("No features to aggregate")
    
    # Stack features
    feature_stack = torch.stack(features, dim=0)  # (N_views, D)
    
    if weights is None:
        # Simple average
        aggregated = feature_stack.mean(dim=0)
    else:
        # Weighted average
        weights = weights.view(-1, 1)  # (N_views, 1)
        aggregated = (feature_stack * weights).sum(dim=0)
    
    return aggregated


def compute_fused_features_for_gaussian(gaussian_id: int,
                                       gaussian_xyz: torch.Tensor,
                                       cameras: List[Camera],
                                       visibility_map: Dict[int, List[int]],
                                       feature_dir: str,
                                       use_visibility_weights: bool = False,
                                       weight_type: str = 'distance') -> Optional[torch.Tensor]:
    """
    Compute fused CLIP features for a single Gaussian.
    
    Args:
        gaussian_id: ID of the Gaussian
        gaussian_xyz: 3D position of the Gaussian
        cameras: List of Camera objects
        visibility_map: Visibility mapping from compute_gaussian_visibility_all_views
        feature_dir: Directory containing CLIP features
        use_visibility_weights: Whether to use visibility-based weighting
        weight_type: Type of visibility weighting
        
    Returns:
        fused_feature: Fused feature vector or None if not visible
    """
    if gaussian_id not in visibility_map:
        return None
        
    visible_indices = visibility_map[gaussian_id]
    
    # Get projected positions for visible views
    projected_positions = []
    for cam_idx in visible_indices:
        from utils.visibility import project_gaussians_to_camera
        visible_mask, proj_xy = project_gaussians_to_camera(
            gaussian_xyz.unsqueeze(0), cameras[cam_idx]
        )
        if visible_mask[0]:
            projected_positions.append((proj_xy[0, 0].item(), proj_xy[0, 1].item()))
    
    # Extract features from all visible views
    features = get_gaussian_clip_features(
        gaussian_id, cameras, visible_indices, projected_positions, feature_dir
    )
    
    if not features:
        return None
    
    # Compute weights if requested
    weights = None
    if use_visibility_weights:
        weights = compute_visibility_weights(
            gaussian_xyz, cameras, visible_indices, weight_type
        )
        # Match weights to extracted features (some views might have missing features)
        if len(weights) != len(features):
            print(f"Warning: Weight count mismatch for Gaussian {gaussian_id}")
            weights = None
    
    # Aggregate features
    fused_feature = aggregate_multiview_features(features, weights)
    
    return fused_feature


def compute_all_fused_features(gaussian_model,
                              cameras: List[Camera],
                              visibility_map: Dict[int, List[int]],
                              feature_dir: str,
                              use_visibility_weights: bool = False,
                              weight_type: str = 'distance',
                              batch_size: int = 1000) -> torch.Tensor:
    """
    Compute fused CLIP features for all Gaussians.
    
    Args:
        gaussian_model: GaussianModel object containing Gaussian positions
        cameras: List of Camera objects
        visibility_map: Visibility mapping
        feature_dir: Directory containing CLIP features
        use_visibility_weights: Whether to use visibility-based weighting
        weight_type: Type of visibility weighting
        batch_size: Batch size for processing
        
    Returns:
        fused_features: Tensor of shape (N, D) with fused features
    """
    num_gaussians = gaussian_model._xyz.shape[0]
    
    # Initialize with zeros (or could use single-view features as fallback)
    feature_dim = 512  # Standard CLIP feature dimension
    fused_features = torch.zeros(num_gaussians, feature_dim, 
                                device=gaussian_model._xyz.device)
    
    # Process in batches for memory efficiency
    for start_idx in range(0, num_gaussians, batch_size):
        end_idx = min(start_idx + batch_size, num_gaussians)
        
        for gaussian_id in range(start_idx, end_idx):
            fused_feat = compute_fused_features_for_gaussian(
                gaussian_id,
                gaussian_model._xyz[gaussian_id],
                cameras,
                visibility_map,
                feature_dir,
                use_visibility_weights,
                weight_type
            )
            
            if fused_feat is not None:
                fused_features[gaussian_id] = fused_feat
                
        print(f"Processed Gaussians {start_idx} to {end_idx}")
    
    return fused_features


def save_fused_features(features: torch.Tensor, output_path: str):
    """Save fused features to .npy file."""
    np.save(output_path, features.cpu().numpy())
    
    
def load_fused_features(path: str) -> torch.Tensor:
    """Load fused features from .npy file."""
    features = np.load(path)
    return torch.from_numpy(features).float()