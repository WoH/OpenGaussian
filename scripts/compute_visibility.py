"""
Script to compute Gaussian visibility across all views for a scene.
Outputs a JSON mapping of gaussian_id to visible camera indices.
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import GaussianModel
from utils.visibility import compute_gaussian_visibility_all_views, save_visibility_json
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    # Set up argument parser
    parser = ArgumentParser(description="Compute Gaussian visibility mapping")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for visibility JSON")
    parser.add_argument("--margin", type=float, default=1.2,
                       help="Frustum margin factor")
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip training cameras")
    parser.add_argument("--skip_test", action="store_true", 
                       help="Skip test cameras")
    
    args = parser.parse_args(sys.argv[1:])
    
    print(f"Computing visibility for model: {args.model_path}")
    
    # Load the trained model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Get Gaussian positions
    gaussian_xyz = gaussians._xyz.detach()
    print(f"Number of Gaussians: {gaussian_xyz.shape[0]}")
    
    # Compute visibility for training cameras
    visibility_maps = {}
    
    if not args.skip_train:
        print("Computing visibility for training cameras...")
        train_cameras = scene.getTrainCameras()
        print(f"Number of training cameras: {len(train_cameras)}")
        
        train_visibility = compute_gaussian_visibility_all_views(
            gaussian_xyz, train_cameras, margin=args.margin
        )
        visibility_maps['train'] = train_visibility
        
        # Print statistics
        visible_counts = [len(v) for v in train_visibility.values()]
        if visible_counts:
            print(f"Train visibility stats:")
            print(f"  - Gaussians with visibility: {len(train_visibility)}")
            print(f"  - Avg views per Gaussian: {sum(visible_counts) / len(visible_counts):.2f}")
            print(f"  - Max views per Gaussian: {max(visible_counts)}")
            print(f"  - Min views per Gaussian: {min(visible_counts)}")
    
    if not args.skip_test:
        print("\nComputing visibility for test cameras...")
        test_cameras = scene.getTestCameras()
        print(f"Number of test cameras: {len(test_cameras)}")
        
        if len(test_cameras) > 0:
            test_visibility = compute_gaussian_visibility_all_views(
                gaussian_xyz, test_cameras, margin=args.margin
            )
            visibility_maps['test'] = test_visibility
            
            # Print statistics
            visible_counts = [len(v) for v in test_visibility.values()]
            if visible_counts:
                print(f"Test visibility stats:")
                print(f"  - Gaussians with visibility: {len(test_visibility)}")
                print(f"  - Avg views per Gaussian: {sum(visible_counts) / len(visible_counts):.2f}")
                print(f"  - Max views per Gaussian: {max(visible_counts)}")
                print(f"  - Min views per Gaussian: {min(visible_counts)}")
    
    # Save visibility mappings
    output_dir = args.output_dir or os.path.join(args.model_path, "visibility")
    os.makedirs(output_dir, exist_ok=True)
    
    for split, visibility_map in visibility_maps.items():
        output_path = os.path.join(output_dir, f"visibility_{split}.json")
        save_visibility_json(visibility_map, output_path)
        print(f"\nSaved {split} visibility to: {output_path}")
        
        # Save a sample for verification
        if visibility_map:
            sample_path = os.path.join(output_dir, f"visibility_{split}_sample.json")
            sample_ids = list(visibility_map.keys())[:10]
            sample_map = {id: visibility_map[id] for id in sample_ids}
            save_visibility_json(sample_map, sample_path)
            print(f"Saved sample to: {sample_path}")


if __name__ == "__main__":
    main()