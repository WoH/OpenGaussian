"""
Training utilities for integrating multi-view CLIP feature fusion.
"""

import torch
import numpy as np
import os
from typing import Optional, List
from utils.feature_fusion import load_fused_features, compute_fused_features_for_gaussian
from utils.visibility import load_visibility_json


class FeatureFusionManager:
    """Manages multi-view CLIP feature fusion during training."""
    
    def __init__(self, args):
        """
        Initialize the feature fusion manager.
        
        Args:
            args: Training arguments containing fusion configuration
        """
        self.use_fused_features = args.use_fused_features
        self.fused_features = None
        self.visibility_map = None
        self.use_visibility_weights = args.use_visibility_weights
        self.weight_type = args.visibility_weight_type
        
        if self.use_fused_features:
            # Load precomputed fused features if available
            if args.fused_features_path and os.path.exists(args.fused_features_path):
                print(f"Loading precomputed fused features from: {args.fused_features_path}")
                self.fused_features = load_fused_features(args.fused_features_path)
                print(f"Loaded fused features shape: {self.fused_features.shape}")
            
            # Load visibility mapping if available
            if args.visibility_json_path and os.path.exists(args.visibility_json_path):
                print(f"Loading visibility mapping from: {args.visibility_json_path}")
                self.visibility_map = load_visibility_json(args.visibility_json_path)
                print(f"Loaded visibility for {len(self.visibility_map)} Gaussians")
    
    def get_fused_features_for_view(self, view, gaussian_model, cameras, feature_dir):
        """
        Get fused features for Gaussians visible in a specific view.
        
        Args:
            view: Current camera view
            gaussian_model: GaussianModel object
            cameras: List of all cameras
            feature_dir: Directory containing CLIP features
            
        Returns:
            fused_features: Tensor of fused features for visible Gaussians
        """
        if not self.use_fused_features:
            return None
            
        # If precomputed features are available, use them
        if self.fused_features is not None:
            return self.fused_features
        
        # Otherwise compute on-the-fly (not recommended for training)
        # This is mainly for debugging/testing
        if self.visibility_map is None:
            print("Warning: No visibility map available for on-the-fly fusion")
            return None
            
        # Here you would implement on-the-fly fusion for visible Gaussians
        # For now, return None to use default single-view features
        return None
    
    def should_use_fused_features(self, iteration, start_feat_iter):
        """
        Determine if fused features should be used at current iteration.
        
        Args:
            iteration: Current training iteration
            start_feat_iter: Iteration to start using features
            
        Returns:
            bool: Whether to use fused features
        """
        return self.use_fused_features and iteration >= start_feat_iter


def get_mask_features_with_fusion(view, mask_bool, rendered_ins_feat, 
                                 fusion_manager, gaussian_model, cameras, 
                                 feature_dir, iteration, start_feat_iter):
    """
    Get mask features with optional multi-view fusion.
    
    Args:
        view: Current camera view
        mask_bool: Boolean masks
        rendered_ins_feat: Rendered instance features
        fusion_manager: FeatureFusionManager instance
        gaussian_model: GaussianModel object
        cameras: List of all cameras
        feature_dir: Directory with CLIP features
        iteration: Current iteration
        start_feat_iter: Start feature iteration
        
    Returns:
        mask_features: Extracted mask features (possibly fused)
    """
    # Check if we should use fused features
    if fusion_manager.should_use_fused_features(iteration, start_feat_iter):
        fused_features = fusion_manager.get_fused_features_for_view(
            view, gaussian_model, cameras, feature_dir
        )
        
        if fused_features is not None:
            # Use fused features instead of single-view features
            # This would require modifying how features are extracted
            # For now, we'll use the standard approach
            pass
    
    # Standard single-view feature extraction
    from utils.opengs_utlis import mask_feature_mean
    return mask_feature_mean(rendered_ins_feat, mask_bool, image_mask=None)


def update_training_config_for_fusion(args):
    """
    Update training configuration for multi-view fusion.
    
    Args:
        args: Training arguments
        
    Returns:
        Updated arguments
    """
    if args.use_fused_features:
        print("\n=== Multi-View CLIP Feature Fusion Configuration ===")
        print(f"Using fused features: {args.use_fused_features}")
        print(f"Fused features path: {args.fused_features_path}")
        print(f"Visibility weighting: {args.use_visibility_weights}")
        if args.use_visibility_weights:
            print(f"Weight type: {args.visibility_weight_type}")
        print(f"Visibility JSON path: {args.visibility_json_path}")
        print("===================================================\n")
        
        # Ensure paths exist if specified
        if args.fused_features_path and not os.path.exists(args.fused_features_path):
            print(f"Warning: Fused features file not found: {args.fused_features_path}")
            
        if args.visibility_json_path and not os.path.exists(args.visibility_json_path):
            print(f"Warning: Visibility JSON file not found: {args.visibility_json_path}")
    
    return args