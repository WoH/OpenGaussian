#!/usr/bin/env python3
"""
Test script for multi-view SAM mask refinement functionality.

Credits:
- Inspired by Gaussian Grouping: https://github.com/lkeab/gaussian-grouping
- Built on OpenGaussian's existing SAM mask processing pipeline
- Uses standard multi-view geometry principles
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from utils.opengs_utlis import detect_view_overlaps, measure_mask_consistency, refine_sam_masks_multiview
from scene.cameras import Camera

def create_dummy_camera(idx, device='cuda'):
    """Create a dummy camera for testing"""
    R = torch.eye(3, device=device)
    T = torch.tensor([idx * 2.0, 0.0, 5.0], device=device)  # Cameras spaced apart
    
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

def test_view_overlap_detection():
    """Test the view overlap detection"""
    print("Testing view overlap detection...")
    
    # Create test cameras
    cameras = [create_dummy_camera(i) for i in range(3)]
    
    # Create dummy depth maps
    depth_maps = {i: torch.ones(480, 640, device='cuda') * 5.0 for i in range(3)}
    
    # Test overlap detection
    overlap_matrix, overlapping_pairs = detect_view_overlaps(cameras, depth_maps)
    
    print(f"Overlap matrix shape: {overlap_matrix.shape}")
    print(f"Found {len(overlapping_pairs)} overlapping pairs")
    print(f"Overlapping pairs: {overlapping_pairs}")
    
    return cameras, depth_maps, overlapping_pairs

def test_mask_consistency():
    """Test mask consistency measurement"""
    print("\nTesting mask consistency measurement...")
    
    # Create dummy masks
    H, W = 100, 100
    num_masks = 3
    
    # Original masks (random squares)
    original_masks = torch.zeros(num_masks, H, W, dtype=torch.bool)
    original_masks[0, 20:40, 20:40] = True
    original_masks[1, 50:70, 30:50] = True
    original_masks[2, 10:30, 60:80] = True
    
    # Reprojected masks (slightly shifted)
    reprojected_masks = torch.zeros(num_masks, H, W, dtype=torch.bool)
    reprojected_masks[0, 22:42, 22:42] = True  # Slightly shifted
    reprojected_masks[1, 50:70, 30:50] = True  # Perfect match
    reprojected_masks[2, 15:35, 65:85] = True  # Shifted
    
    valid_mask = torch.ones(H, W, dtype=torch.bool)
    
    consistency_scores, iou_scores = measure_mask_consistency(
        original_masks, reprojected_masks, valid_mask)
    
    print(f"Consistency scores: {consistency_scores}")
    print(f"IoU scores: {iou_scores}")

def test_full_refinement():
    """Test the full multi-view refinement pipeline"""
    print("\nTesting full multi-view refinement...")
    
    cameras, depth_maps, overlapping_pairs = test_view_overlap_detection()
    
    if not overlapping_pairs:
        print("No overlapping pairs found, creating synthetic overlap...")
        overlapping_pairs = [(0, 1), (1, 2)]
    
    # Create dummy SAM masks for each view
    H, W = 240, 320
    num_masks = 2
    
    sam_masks_dict = {}
    for view_idx in range(3):
        masks = torch.zeros(num_masks, H, W, dtype=torch.bool, device='cuda')
        # Create different masks for each view to test refinement
        masks[0, 50:100, 50:100] = True
        masks[1, 120:170, 150:200] = True
        
        # Add some noise/variation per view
        if view_idx == 1:
            masks[0, 55:105, 55:105] = True  # Slightly larger
        elif view_idx == 2:
            masks[1, 115:165, 145:195] = True  # Slightly shifted
            
        sam_masks_dict[view_idx] = masks
    
    # Test refinement
    try:
        refined_masks = refine_sam_masks_multiview(
            sam_masks_dict, cameras, depth_maps, 
            max_iterations=2, consistency_threshold=0.3)
        
        print("Multi-view refinement completed successfully!")
        print(f"Original masks keys: {list(sam_masks_dict.keys())}")
        print(f"Refined masks keys: {list(refined_masks.keys())}")
        
        # Compare before/after for first view
        original_sum = sam_masks_dict[0].sum().item()
        refined_sum = refined_masks[0].sum().item()
        print(f"View 0 - Original mask pixels: {original_sum}, Refined: {refined_sum}")
        
    except Exception as e:
        print(f"Error during refinement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Multi-view SAM mask refinement test")
    print("=" * 50)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, some tests may fail")
    
    try:
        test_view_overlap_detection()
        test_mask_consistency()
        test_full_refinement()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()