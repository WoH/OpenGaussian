"""
Script to extract and fuse multi-view CLIP features for all Gaussians.
Uses precomputed visibility mappings and existing CLIP features.
"""

import argparse
import os
import sys
import torch
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scene import Scene
from gaussian_renderer import GaussianModel
from utils.visibility import load_visibility_json
from utils.feature_fusion import compute_all_fused_features, save_fused_features
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    # Set up argument parser
    parser = ArgumentParser(description="Extract fused multi-view CLIP features")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--visibility_json", type=str, required=True,
                       help="Path to visibility JSON file")
    parser.add_argument("--feature_dir", type=str, required=True,
                       help="Directory containing CLIP features")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for fused features")
    parser.add_argument("--use_visibility_weights", action="store_true",
                       help="Use visibility-based weighting")
    parser.add_argument("--weight_type", type=str, default="distance",
                       choices=["uniform", "distance", "angle", "area"],
                       help="Type of visibility weighting")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for processing")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test"],
                       help="Which camera split to use")
    
    args = parser.parse_args(sys.argv[1:])
    
    print(f"Extracting fused features for model: {args.model_path}")
    print(f"Using visibility from: {args.visibility_json}")
    print(f"CLIP features from: {args.feature_dir}")
    
    # Load the trained model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # Load visibility mapping
    visibility_map = load_visibility_json(args.visibility_json)
    print(f"Loaded visibility for {len(visibility_map)} Gaussians")
    
    # Get cameras
    if args.split == "train":
        cameras = scene.getTrainCameras()
    else:
        cameras = scene.getTestCameras()
    print(f"Using {len(cameras)} {args.split} cameras")
    
    # Compute fused features
    print(f"\nComputing fused features...")
    print(f"  - Visibility weighting: {args.use_visibility_weights}")
    if args.use_visibility_weights:
        print(f"  - Weight type: {args.weight_type}")
    
    fused_features = compute_all_fused_features(
        gaussians,
        cameras,
        visibility_map,
        args.feature_dir,
        use_visibility_weights=args.use_visibility_weights,
        weight_type=args.weight_type,
        batch_size=args.batch_size
    )
    
    print(f"\nFused features shape: {fused_features.shape}")
    
    # Save fused features
    if args.output_path is None:
        output_dir = os.path.join(args.model_path, "fused_features")
        os.makedirs(output_dir, exist_ok=True)
        weight_suffix = f"_{args.weight_type}" if args.use_visibility_weights else "_uniform"
        output_path = os.path.join(output_dir, f"fused_features_{args.split}{weight_suffix}.npy")
    else:
        output_path = args.output_path
    
    save_fused_features(fused_features, output_path)
    print(f"\nSaved fused features to: {output_path}")
    
    # Save some statistics
    stats = {
        "num_gaussians": fused_features.shape[0],
        "feature_dim": fused_features.shape[1],
        "num_cameras": len(cameras),
        "visibility_weighting": args.use_visibility_weights,
        "weight_type": args.weight_type if args.use_visibility_weights else "uniform",
        "split": args.split
    }
    
    stats_path = output_path.replace(".npy", "_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")
    
    # Compute feature statistics
    nonzero_features = fused_features[fused_features.norm(dim=1) > 0]
    if len(nonzero_features) > 0:
        print(f"\nFeature statistics:")
        print(f"  - Gaussians with features: {len(nonzero_features)}")
        print(f"  - Mean feature norm: {nonzero_features.norm(dim=1).mean():.4f}")
        print(f"  - Std feature norm: {nonzero_features.norm(dim=1).std():.4f}")


if __name__ == "__main__":
    main()