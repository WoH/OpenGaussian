"""
Visibility computation module for multi-view CLIP feature fusion.
Determines which 3D Gaussians are visible in which camera views.
"""

import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from scene.cameras import Camera


def project_gaussians_to_camera(gaussians_xyz: torch.Tensor, 
                                camera: Camera,
                                margin: float = 1.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussian centers to camera view and check visibility.
    
    Args:
        gaussians_xyz: Tensor of shape (N, 3) with Gaussian centers
        camera: Camera object containing projection matrices
        margin: Margin factor for frustum culling (1.2 = 20% margin)
        
    Returns:
        visible_mask: Boolean tensor of shape (N,) indicating visibility
        projected_xy: Tensor of shape (N, 2) with normalized projected coordinates
    """
    # Transform to camera space
    xyz_homo = torch.cat([gaussians_xyz, torch.ones(gaussians_xyz.shape[0], 1, device=gaussians_xyz.device)], dim=1)
    cam_xyz = torch.matmul(camera.world_view_transform, xyz_homo.T).T
    
    # Check if points are in front of camera
    in_front = cam_xyz[:, 2] > camera.znear
    
    # Project to screen space
    proj_xyz = torch.matmul(camera.full_proj_transform, xyz_homo.T).T
    proj_xy = proj_xyz[:, :2] / proj_xyz[:, 3:4]
    
    # Normalize to [0, 1] range
    norm_x = (proj_xy[:, 0] + 1) * 0.5
    norm_y = (proj_xy[:, 1] + 1) * 0.5
    
    # Check if within frustum (with margin)
    in_frustum = (norm_x >= -margin + 1) & (norm_x <= margin) & \
                 (norm_y >= -margin + 1) & (norm_y <= margin)
    
    visible_mask = in_front & in_frustum
    
    return visible_mask, torch.stack([norm_x, norm_y], dim=1)


def compute_gaussian_visibility_all_views(gaussians_xyz: torch.Tensor,
                                         cameras: List[Camera],
                                         margin: float = 1.2) -> Dict[int, List[int]]:
    """
    Compute visibility of all Gaussians across all camera views.
    
    Args:
        gaussians_xyz: Tensor of shape (N, 3) with Gaussian centers
        cameras: List of Camera objects
        margin: Margin factor for frustum culling
        
    Returns:
        visibility_map: Dictionary mapping gaussian_id to list of visible camera indices
    """
    visibility_map = {}
    
    for gaussian_id in range(gaussians_xyz.shape[0]):
        visible_cameras = []
        
        for cam_idx, camera in enumerate(cameras):
            # Check single Gaussian visibility
            gaussian_pos = gaussians_xyz[gaussian_id:gaussian_id+1]
            visible_mask, _ = project_gaussians_to_camera(gaussian_pos, camera, margin)
            
            if visible_mask[0]:
                visible_cameras.append(cam_idx)
        
        if visible_cameras:
            visibility_map[gaussian_id] = visible_cameras
    
    return visibility_map


def compute_visibility_weights(gaussian_xyz: torch.Tensor,
                              cameras: List[Camera],
                              visible_indices: List[int],
                              weight_type: str = 'distance') -> torch.Tensor:
    """
    Compute visibility weights for a Gaussian across its visible views.
    
    Args:
        gaussian_xyz: Tensor of shape (3,) with single Gaussian center
        cameras: List of all Camera objects
        visible_indices: List of camera indices where Gaussian is visible
        weight_type: Type of weighting ('uniform', 'distance', 'angle', 'area')
        
    Returns:
        weights: Normalized weights tensor of shape (len(visible_indices),)
    """
    if not visible_indices:
        return torch.tensor([])
    
    weights = []
    
    for cam_idx in visible_indices:
        camera = cameras[cam_idx]
        
        if weight_type == 'uniform':
            weight = 1.0
            
        elif weight_type == 'distance':
            # Weight inversely proportional to squared distance
            cam_center = camera.camera_center
            distance = torch.norm(gaussian_xyz - cam_center)
            weight = 1.0 / (distance ** 2 + 1e-6)
            
        elif weight_type == 'angle':
            # Weight based on viewing angle (dot product with camera forward)
            cam_center = camera.camera_center
            view_dir = gaussian_xyz - cam_center
            view_dir = view_dir / (torch.norm(view_dir) + 1e-6)
            
            # Camera forward direction (negative z in camera space)
            cam_forward = -camera.world_view_transform[2, :3]
            
            # Cosine similarity (higher weight for more direct viewing angles)
            weight = torch.abs(torch.dot(view_dir, cam_forward))
            
        elif weight_type == 'area':
            # Weight based on projected area (requires Gaussian scale info)
            # For now, use distance as proxy
            cam_center = camera.camera_center
            distance = torch.norm(gaussian_xyz - cam_center)
            weight = 1.0 / (distance + 1e-6)
            
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
            
        weights.append(weight)
    
    weights = torch.tensor(weights, device=gaussian_xyz.device)
    
    # Normalize weights to sum to 1
    weights = weights / (weights.sum() + 1e-6)
    
    return weights


def save_visibility_json(visibility_map: Dict[int, List[int]], 
                        output_path: str):
    """Save visibility mapping to JSON file."""
    # Convert to regular Python types for JSON serialization
    json_map = {
        str(k): [int(v) for v in vals] 
        for k, vals in visibility_map.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(json_map, f, indent=2)
        
        
def load_visibility_json(json_path: str) -> Dict[int, List[int]]:
    """Load visibility mapping from JSON file."""
    with open(json_path, 'r') as f:
        json_map = json.load(f)
    
    # Convert string keys back to integers
    visibility_map = {
        int(k): vals 
        for k, vals in json_map.items()
    }
    
    return visibility_map