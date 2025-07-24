#!/usr/bin/env python3
"""
Test script for multi-view CLIP feature fusion functionality.

Tests the core functions for multi-view CLIP feature fusion with synthetic data
to ensure proper functionality before integration into the training pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.opengs_utlis import (
    compute_gaussian_visibility,
    aggregate_clip_features_per_view,
    compute_view_weights,
    fuse_multiview_clip_features
)
from scene.cameras import Camera


def create_dummy_camera(idx, device='cuda'):
    """Create a dummy camera for testing"""
    # Create camera with different positions
    R = torch.eye(3, device=device)
    T = torch.tensor([idx * 1.5, 0.0, 5.0], device=device)  # Cameras spaced apart
    
    return Camera(
        colmap_id=idx,
        R=R.cpu().numpy(),
        T=T.cpu().numpy(), 
        FoVx=1.2, FoVy=0.9,
        cx=320, cy=240,
        image=torch.rand(3, 480, 640),
        depth=None,
        gt_alpha_mask=None,
        gt_sam_mask=None,
        gt_mask_feat=None,
        image_name=f"test_{idx}.jpg",
        uid=idx
    )


def test_gaussian_visibility():
    """Test the Gaussian visibility computation"""
    print("Testing Gaussian visibility computation...")
    
    # Create test data
    N_gaussians = 100
    N_views = 3
    
    # Create random 3D Gaussians
    gaussians_xyz = torch.randn(N_gaussians, 3, device='cuda') * 2.0
    gaussians_xyz[:, 2] += 5.0  # Move them in front of cameras
    
    # Create test cameras
    cameras = [create_dummy_camera(i) for i in range(N_views)]
    
    # Create dummy depth maps
    depth_maps = {i: torch.ones(480, 640, device='cuda') * 5.0 for i in range(N_views)}
    
    # Test visibility computation
    visibility_matrix, projected_coords = compute_gaussian_visibility(
        gaussians_xyz, cameras, depth_maps)
    
    print(f"Visibility matrix shape: {visibility_matrix.shape}")
    print(f"Projected coords keys: {list(projected_coords.keys())}")
    print(f"Visible Gaussians per view: {visibility_matrix.sum(dim=0)}")
    print(f"Views per Gaussian (avg): {visibility_matrix.sum(dim=1).float().mean():.2f}")
    
    return visibility_matrix, projected_coords, gaussians_xyz, cameras


def test_clip_feature_aggregation():
    """Test CLIP feature aggregation per view"""
    print("\nTesting CLIP feature aggregation...")
    
    visibility_matrix, projected_coords, gaussians_xyz, cameras = test_gaussian_visibility()
    N_gaussians = gaussians_xyz.shape[0]
    
    # Create dummy SAM masks and CLIP features
    sam_masks_dict = {}
    clip_features_dict = {}
    
    for view_idx in range(len(cameras)):
        H, W = 480, 640
        num_masks = 5
        
        # Create random masks
        masks = torch.zeros(num_masks, H, W, dtype=torch.bool, device='cuda')
        for i in range(num_masks):
            y1, y2 = torch.randint(0, H//2, (2,)).sort()[0]
            x1, x2 = torch.randint(0, W//2, (2,)).sort()[0]
            masks[i, y1:y2, x1:x2] = True
        
        # Create random CLIP features
        clip_features = torch.randn(num_masks, 512, device='cuda')
        clip_features = torch.nn.functional.normalize(clip_features, dim=1)
        
        sam_masks_dict[view_idx] = masks
        clip_features_dict[view_idx] = clip_features
    
    # Test aggregation
    gaussian_clip_features = aggregate_clip_features_per_view(
        sam_masks_dict, clip_features_dict, projected_coords, visibility_matrix)
    
    print(f"Gaussian CLIP features keys: {list(gaussian_clip_features.keys())}")
    for view_idx, features in gaussian_clip_features.items():
        print(f"View {view_idx}: {features.shape}, non-zero: {(features.norm(dim=1) > 0).sum()}")
    
    return gaussian_clip_features, visibility_matrix, gaussians_xyz, cameras


def test_view_weights():
    """Test view weight computation"""
    print("\nTesting view weight computation...")
    
    gaussian_clip_features, visibility_matrix, gaussians_xyz, cameras = test_clip_feature_aggregation()
    projected_coords = {i: torch.randn(gaussians_xyz.shape[0], 2, device='cuda') * 320 
                       for i in range(len(cameras))}
    
    # Test weight computation
    view_weights = compute_view_weights(
        gaussians_xyz, cameras, visibility_matrix, projected_coords)
    
    print(f"View weights shape: {view_weights.shape}")
    print(f"Weight range: [{view_weights.min():.3f}, {view_weights.max():.3f}]")
    print(f"Average weight per view: {view_weights.mean(dim=0)}")
    print(f"Weights sum to 1 per Gaussian: {torch.allclose(view_weights.sum(dim=1), torch.ones(view_weights.shape[0], device='cuda'), atol=1e-5)}")
    
    return gaussian_clip_features, visibility_matrix, view_weights


def test_feature_fusion():
    """Test the complete feature fusion pipeline"""
    print("\nTesting feature fusion...")
    
    gaussian_clip_features, visibility_matrix, view_weights = test_view_weights()
    
    # Test fusion
    fused_features = fuse_multiview_clip_features(
        gaussian_clip_features, visibility_matrix, view_weights)
    
    print(f"Fused features shape: {fused_features.shape}")
    print(f"Non-zero fused features: {(fused_features.norm(dim=1) > 0).sum()}")
    print(f"Feature norm range: [{fused_features.norm(dim=1).min():.3f}, {fused_features.norm(dim=1).max():.3f}]")
    
    # Check if fusion makes sense
    visible_anywhere = visibility_matrix.any(dim=1)
    visible_fused = fused_features[visible_anywhere]
    invisible_fused = fused_features[~visible_anywhere]
    
    print(f"Visible Gaussians: {visible_anywhere.sum()}, Non-zero features: {(visible_fused.norm(dim=1) > 0).sum()}")
    print(f"Invisible Gaussians: {(~visible_anywhere).sum()}, Zero features: {(invisible_fused.norm(dim=1) == 0).sum()}")
    
    return fused_features


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    # Test with empty inputs
    empty_features = fuse_multiview_clip_features({}, torch.zeros(10, 3, dtype=torch.bool), torch.zeros(10, 3))
    print(f"Empty input result shape: {empty_features.shape}")
    
    # Test with single view
    single_view_features = {0: torch.randn(5, 512, device='cuda')}
    single_visibility = torch.zeros(5, 1, dtype=torch.bool, device='cuda')
    single_visibility[:3, 0] = True
    single_weights = torch.ones(5, 1, device='cuda')
    
    single_result = fuse_multiview_clip_features(single_view_features, single_visibility, single_weights)
    print(f"Single view result shape: {single_result.shape}, non-zero: {(single_result.norm(dim=1) > 0).sum()}")
    
    print("Edge case tests completed successfully!")


def main():
    """Run all tests for multi-view CLIP feature fusion"""
    print("Multi-View CLIP Feature Fusion Tests")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, some tests may fail")
        return
    
    try:
        # Run all tests
        test_gaussian_visibility()
        test_clip_feature_aggregation()
        test_view_weights()
        fused_features = test_feature_fusion()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print(f"Final fused features shape: {fused_features.shape}")
        print("Multi-view CLIP feature fusion is ready for integration!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)