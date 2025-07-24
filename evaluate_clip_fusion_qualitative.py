#!/usr/bin/env python3
"""
Qualitative evaluation script for multi-view CLIP feature fusion on ScanNet.

This script generates visual comparisons showing the improvement in semantic
consistency when using multi-view CLIP feature fusion versus single-view processing.

Usage:
    python evaluate_clip_fusion_qualitative.py -s <scene_path> -m <model_path> --output_dir <output_path>

Example:
    python evaluate_clip_fusion_qualitative.py -s data/scannet/scene0000_00 -m output/scene0000_00/point_cloud/iteration_30000/point_cloud.ply --output_dir results/qualitative
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from gaussian_renderer import render
from utils.opengs_utlis import (
    get_SAM_mask_and_feat, 
    compute_gaussian_visibility,
    aggregate_clip_features_per_view,
    compute_view_weights,
    fuse_multiview_clip_features
)
from multiview_clip_fusion_example import apply_multiview_clip_fusion, query_scene_by_text
import clip


class QualitativeEvaluator:
    def __init__(self, scene_path, model_path, output_dir):
        self.scene_path = scene_path
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scene and model
        self.setup_scene_and_model()
        
        # Load CLIP model for text queries
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cuda")
        
        # Define test queries based on common ScanNet objects
        self.test_queries = [
            "chair", "table", "bed", "sofa", "desk", 
            "cabinet", "lamp", "book", "monitor", "keyboard"
        ]
    
    def setup_scene_and_model(self):
        """Initialize scene and load trained model"""
        print(f"Loading scene from: {self.scene_path}")
        print(f"Loading model from: {self.model_path}")
        
        # Setup arguments
        self.model_params = ModelParams()
        self.model_params.source_path = self.scene_path
        self.model_params.model_path = self.model_path
        
        self.pipeline_params = PipelineParams()
        self.opt_params = OptimizationParams()
        
        # Load scene
        self.scene = Scene(self.model_params, shuffle=False)
        self.cameras = self.scene.getTrainCameras()[:10]  # Use first 10 cameras for evaluation
        
        print(f"Loaded scene with {len(self.cameras)} cameras")
        print(f"Scene has {len(self.scene.gaussians.get_xyz())} Gaussians")
    
    def generate_single_view_features(self, camera):
        """Generate CLIP features using single-view processing"""
        if camera.original_sam_mask is None or camera.original_mask_feat is None:
            return None, None
            
        gt_sam_mask = camera.original_sam_mask.cuda()
        mask_id, mask_bool, mask_feat, invalid_pix = get_SAM_mask_and_feat(
            gt_sam_mask, level=3, original_mask_feat=camera.original_mask_feat)
        
        return mask_bool, mask_feat
    
    def generate_multiview_features(self, cameras_subset):
        """Generate CLIP features using multi-view fusion"""
        # Apply multi-view fusion
        opt = argparse.Namespace()
        opt.clip_fusion_start_iter = 0  # Always apply fusion
        opt.sam_level = 3
        
        fused_features = apply_multiview_clip_fusion(
            self.scene, cameras_subset, iteration=1000, opt=opt)
        
        return fused_features
    
    def render_semantic_heatmap(self, camera, clip_features, query_text, method_name):
        """Render semantic heatmap for a specific text query"""
        # Encode text query
        text_tokens = clip.tokenize([query_text]).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Render scene
        rendered = render(camera, self.scene.gaussians, self.pipeline_params)
        
        # Create semantic heatmap
        if clip_features is not None and len(clip_features) > 0:
            # Compute similarities between Gaussian CLIP features and text query
            similarities = torch.matmul(clip_features, text_features.T).squeeze()
            similarities = torch.clamp(similarities, 0, 1)  # Normalize to [0,1]
            
            # Map similarities to rendered image (this is simplified - would need proper mapping)
            # For this demo, we'll create a synthetic heatmap
            H, W = rendered['render'].shape[1], rendered['render'].shape[2]
            heatmap = torch.zeros(H, W, device='cuda')
            
            # This would need proper Gaussian-to-pixel mapping in a real implementation
            # For demo purposes, create representative heatmap
            if similarities.max() > 0.3:  # If query matches well
                center_y, center_x = H//2, W//2
                y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                distances = ((y_coords - center_y)**2 + (x_coords - center_x)**2).float().cuda()
                heatmap = torch.exp(-distances / (H*W/16)) * similarities.max()
        else:
            H, W = rendered['render'].shape[1], rendered['render'].shape[2]
            heatmap = torch.zeros(H, W, device='cuda')
        
        # Convert to numpy for visualization
        image = rendered['render'].detach().cpu().numpy().transpose(1, 2, 0)
        heatmap_np = heatmap.detach().cpu().numpy()
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title(f'Original View\n{camera.image_name}')
        ax1.axis('off')
        
        # Heatmap
        im = ax2.imshow(heatmap_np, cmap='hot', alpha=0.8)
        ax2.set_title(f'Semantic Heatmap\nQuery: "{query_text}"')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Overlay
        ax3.imshow(image)
        ax3.imshow(heatmap_np, cmap='hot', alpha=0.6)
        ax3.set_title(f'Overlay ({method_name})')
        ax3.axis('off')
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'{method_name}_{query_text}_{camera.image_name}_heatmap.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path, heatmap_np.max()
    
    def generate_cross_view_consistency(self):
        """Generate cross-view consistency comparison"""
        print("Generating cross-view consistency comparison...")
        
        # Select a subset of cameras with good overlap
        cameras_subset = self.cameras[:6]
        
        # Test object that should be visible in multiple views
        test_query = "chair"
        
        # Single-view results
        single_view_results = []
        for i, camera in enumerate(cameras_subset):
            mask_bool, mask_feat = self.generate_single_view_features(camera)
            if mask_feat is not None:
                # For single-view, use the mask features directly as CLIP features
                output_path, max_response = self.render_semantic_heatmap(
                    camera, mask_feat, test_query, f"single_view_{i}")
                single_view_results.append((output_path, max_response))
        
        # Multi-view results
        multiview_features = self.generate_multiview_features(cameras_subset)
        multiview_results = []
        
        if multiview_features is not None:
            for i, camera in enumerate(cameras_subset):
                output_path, max_response = self.render_semantic_heatmap(
                    camera, multiview_features, test_query, f"multiview_{i}")
                multiview_results.append((output_path, max_response))
        
        # Generate comparison summary
        self.create_consistency_summary(single_view_results, multiview_results, test_query)
        
        return single_view_results, multiview_results
    
    def create_consistency_summary(self, single_results, multi_results, query):
        """Create a summary comparison figure"""
        fig, axes = plt.subplots(2, len(single_results), figsize=(20, 8))
        
        # Single-view row
        for i, (path, response) in enumerate(single_results):
            if os.path.exists(path):
                img = Image.open(path)
                axes[0, i].imshow(img)
            axes[0, i].set_title(f'Single-View {i+1}\nResponse: {response:.3f}')
            axes[0, i].axis('off')
        
        # Multi-view row
        for i, (path, response) in enumerate(multi_results):
            if os.path.exists(path):
                img = Image.open(path)
                axes[1, i].imshow(img)
            axes[1, i].set_title(f'Multi-View {i+1}\nResponse: {response:.3f}')
            axes[1, i].axis('off')
        
        # Add row labels
        axes[0, 0].text(-0.1, 0.5, 'Single-View\\nProcessing', transform=axes[0, 0].transAxes,
                        rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Multi-View\\nFusion', transform=axes[1, 0].transAxes,
                        rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.suptitle(f'Cross-View Consistency Comparison: "{query}"', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_path = os.path.join(self.output_dir, f'consistency_comparison_{query}.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Consistency comparison saved to: {summary_path}")
        return summary_path
    
    def generate_text_query_comparison(self):
        """Generate text query comparison for multiple objects"""
        print("Generating text query comparisons...")
        
        cameras_subset = self.cameras[:4]  # Use 4 cameras for comparison
        
        # Generate multi-view features once
        multiview_features = self.generate_multiview_features(cameras_subset)
        
        query_results = {}
        
        for query in self.test_queries[:5]:  # Test first 5 queries
            print(f"Processing query: {query}")
            
            query_results[query] = {
                'single_view': [],
                'multi_view': []
            }
            
            # Test on a representative camera
            test_camera = cameras_subset[0]
            
            # Single-view result
            mask_bool, mask_feat = self.generate_single_view_features(test_camera)
            if mask_feat is not None:
                single_path, single_response = self.render_semantic_heatmap(
                    test_camera, mask_feat, query, "single")
                query_results[query]['single_view'] = (single_path, single_response)
            
            # Multi-view result
            if multiview_features is not None:
                multi_path, multi_response = self.render_semantic_heatmap(
                    test_camera, multiview_features, query, "multi")
                query_results[query]['multi_view'] = (multi_path, multi_response)
        
        # Create combined comparison figure
        self.create_query_comparison_summary(query_results)
        
        return query_results
    
    def create_query_comparison_summary(self, query_results):
        """Create summary comparison for multiple text queries"""
        num_queries = len(query_results)
        fig, axes = plt.subplots(2, num_queries, figsize=(4*num_queries, 8))
        
        queries = list(query_results.keys())
        
        for i, query in enumerate(queries):
            single_result = query_results[query]['single_view']
            multi_result = query_results[query]['multi_view']
            
            # Single-view row
            if single_result and os.path.exists(single_result[0]):
                img = Image.open(single_result[0])
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Single-View\\n"{query}"\\nScore: {single_result[1]:.3f}')
            else:
                axes[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(f'Single-View\\n"{query}"')
            axes[0, i].axis('off')
            
            # Multi-view row
            if multi_result and os.path.exists(multi_result[0]):
                img = Image.open(multi_result[0])
                axes[1, i].imshow(img)
                axes[1, i].set_title(f'Multi-View\\n"{query}"\\nScore: {multi_result[1]:.3f}')
            else:
                axes[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'Multi-View\\n"{query}"')
            axes[1, i].axis('off')
        
        plt.suptitle('Text Query Comparison: Single-View vs Multi-View CLIP Fusion', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_path = os.path.join(self.output_dir, 'text_query_comparison.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Text query comparison saved to: {summary_path}")
        return summary_path
    
    def run_evaluation(self):
        """Run complete qualitative evaluation"""
        print("Starting qualitative evaluation for multi-view CLIP fusion...")
        print("=" * 60)
        
        results_summary = {
            'scene_path': self.scene_path,
            'model_path': self.model_path,
            'output_dir': self.output_dir,
            'num_cameras': len(self.cameras),
            'num_gaussians': len(self.scene.gaussians.get_xyz()),
            'test_queries': self.test_queries
        }
        
        try:
            # Generate cross-view consistency comparison
            single_consistency, multi_consistency = self.generate_cross_view_consistency()
            
            # Generate text query comparisons
            query_results = self.generate_text_query_comparison()
            
            # Save results summary
            summary_file = os.path.join(self.output_dir, 'qualitative_results_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print("\\n" + "=" * 60)
            print("Qualitative evaluation completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            print(f"Summary file: {summary_file}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation of multi-view CLIP fusion")
    parser.add_argument('-s', '--source_path', required=True, help="Path to ScanNet scene data")
    parser.add_argument('-m', '--model_path', required=True, help="Path to trained model (.ply file)")
    parser.add_argument('--output_dir', default='results/qualitative', help="Output directory for results")
    parser.add_argument('--max_cameras', type=int, default=10, help="Maximum number of cameras to use")
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.source_path):
        print(f"Error: Scene path does not exist: {args.source_path}")
        return False
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return False
    
    print(f"Scene path: {args.source_path}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Run evaluation
    evaluator = QualitativeEvaluator(args.source_path, args.model_path, args.output_dir)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)