#!/usr/bin/env python3
"""
Benchmark script for multi-view SAM mask refinement.

Produces before/after comparison metrics and visualizations to evaluate
the impact of multi-view refinement on mask quality.

Credits:
- Inspired by Gaussian Grouping: https://github.com/lkeab/gaussian-grouping  
- Built on OpenGaussian's existing SAM mask processing pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.opengs_utlis import get_SAM_mask_and_feat, refine_sam_masks_multiview, detect_view_overlaps
from scene import Scene
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render


class MultiViewSAMBenchmark:
    def __init__(self, scene_path, output_dir="benchmark_results"):
        self.scene_path = scene_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize scene
        self.model_params = ModelParams()
        self.model_params.source_path = scene_path
        self.pipeline_params = PipelineParams()
        self.opt_params = OptimizationParams()
        
        self.scene = Scene(self.model_params, shuffle=False)
        self.cameras = self.scene.getTrainCameras()
        
        self.results = {
            'before': {'metrics': [], 'timings': []},
            'after': {'metrics': [], 'timings': []}
        }
    
    def compute_mask_metrics(self, masks, name=""):
        """Compute various metrics for mask quality assessment"""
        metrics = {}
        
        if len(masks) == 0:
            return metrics
            
        # Convert dict to list if needed
        if isinstance(masks, dict):
            mask_list = list(masks.values())
        else:
            mask_list = [masks] if not isinstance(masks, list) else masks
            
        total_pixels = 0
        total_masks = 0
        boundary_sharpness = []
        
        for view_masks in mask_list:
            if isinstance(view_masks, torch.Tensor):
                num_masks, H, W = view_masks.shape
                total_masks += num_masks
                total_pixels += view_masks.sum().item()
                
                # Compute boundary sharpness (edge strength)
                for mask_idx in range(num_masks):
                    mask = view_masks[mask_idx].float()
                    # Sobel edge detection
                    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                         dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
                    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                         dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
                    
                    edges_x = torch.conv2d(mask.unsqueeze(0).unsqueeze(0), sobel_x, padding=1)
                    edges_y = torch.conv2d(mask.unsqueeze(0).unsqueeze(0), sobel_y, padding=1)
                    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
                    
                    if edge_magnitude.sum() > 0:
                        boundary_sharpness.append(edge_magnitude.mean().item())
        
        metrics.update({
            'total_masks': total_masks,
            'total_pixels': total_pixels,
            'avg_boundary_sharpness': np.mean(boundary_sharpness) if boundary_sharpness else 0.0,
            'mask_coverage_ratio': total_pixels / (len(mask_list) * H * W) if mask_list else 0.0
        })
        
        return metrics
    
    def benchmark_single_view_processing(self, sample_views=10):
        """Benchmark original single-view SAM mask processing"""
        print("Benchmarking single-view SAM processing...")
        
        before_masks = {}
        processing_times = []
        
        # Sample a subset of views for benchmarking
        sample_cameras = self.cameras[:sample_views] if len(self.cameras) > sample_views else self.cameras
        
        for idx, camera in enumerate(sample_cameras):
            if camera.original_sam_mask is None:
                continue
                
            start_time = time.time()
            
            # Original single-view processing
            gt_sam_mask = camera.original_sam_mask.cuda()
            mask_id, mask_bool, invalid_pix = get_SAM_mask_and_feat(
                gt_sam_mask, level=3, filter_th=50
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            before_masks[idx] = mask_bool
            
            print(f"View {idx}: {mask_bool.shape[0]} masks, {processing_time:.4f}s")
        
        # Compute metrics
        metrics = self.compute_mask_metrics(before_masks, "single_view")
        metrics['avg_processing_time'] = np.mean(processing_times)
        metrics['total_processing_time'] = sum(processing_times)
        
        self.results['before']['metrics'] = metrics
        self.results['before']['timings'] = processing_times
        
        return before_masks, metrics
    
    def benchmark_multiview_refinement(self, before_masks, sample_views=10):
        """Benchmark multi-view refinement processing"""
        print("Benchmarking multi-view SAM refinement...")
        
        sample_cameras = self.cameras[:sample_views] if len(self.cameras) > sample_views else self.cameras
        
        # Create depth maps (using dummy data for benchmark)
        depth_maps = {}
        for idx, camera in enumerate(sample_cameras):
            if idx in before_masks:
                H, W = before_masks[idx].shape[1], before_masks[idx].shape[2]
                depth_maps[idx] = torch.ones(H, W, device='cuda') * 5.0
        
        start_time = time.time()
        
        # Apply multi-view refinement
        try:
            refined_masks = refine_sam_masks_multiview(
                before_masks, 
                [sample_cameras[i] for i in before_masks.keys()],
                depth_maps,
                max_iterations=3,
                consistency_threshold=0.5
            )
            
            processing_time = time.time() - start_time
            
            # Compute metrics
            metrics = self.compute_mask_metrics(refined_masks, "multiview")
            metrics['total_processing_time'] = processing_time
            metrics['avg_processing_time'] = processing_time / len(refined_masks)
            
            self.results['after']['metrics'] = metrics
            self.results['after']['timings'] = [processing_time]
            
            print(f"Multi-view refinement: {len(refined_masks)} views, {processing_time:.4f}s total")
            
            return refined_masks, metrics
            
        except Exception as e:
            print(f"Multi-view refinement failed: {e}")
            return before_masks, self.results['before']['metrics']
    
    def compute_comparison_metrics(self, before_masks, after_masks):
        """Compute before/after comparison metrics"""
        comparison = {}
        
        # Pixel difference analysis
        total_pixel_diff = 0
        total_pixels = 0
        consistency_improvements = []
        
        for view_idx in before_masks.keys():
            if view_idx in after_masks:
                before_mask = before_masks[view_idx]
                after_mask = after_masks[view_idx]
                
                # Pixel-level differences
                diff = (before_mask != after_mask).sum().item()
                total_pixel_diff += diff
                total_pixels += before_mask.numel()
                
                # Mask count consistency
                before_count = before_mask.shape[0]
                after_count = after_mask.shape[0]
                consistency_improvements.append(abs(before_count - after_count))
        
        comparison.update({
            'pixel_change_ratio': total_pixel_diff / total_pixels if total_pixels > 0 else 0.0,
            'avg_mask_count_change': np.mean(consistency_improvements) if consistency_improvements else 0.0,
            'views_processed': len(before_masks),
            'speedup_ratio': (self.results['before']['metrics']['total_processing_time'] / 
                            self.results['after']['metrics']['total_processing_time']) 
                           if self.results['after']['metrics']['total_processing_time'] > 0 else 1.0
        })
        
        return comparison
    
    def generate_report(self, before_masks, after_masks, comparison_metrics):
        """Generate detailed benchmark report"""
        
        report = {
            'scene_path': str(self.scene_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'views_processed': len(before_masks),
                'total_cameras': len(self.cameras)
            },
            'single_view_results': self.results['before']['metrics'],
            'multiview_results': self.results['after']['metrics'],
            'comparison': comparison_metrics
        }
        
        # Save JSON report
        report_file = self.output_dir / 'benchmark_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate text summary
        summary_file = self.output_dir / 'benchmark_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Multi-View SAM Mask Refinement Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scene: {self.scene_path}\n")
            f.write(f"Views processed: {comparison_metrics['views_processed']}\n")
            f.write(f"Total cameras: {len(self.cameras)}\n\n")
            
            f.write("BEFORE (Single-View Processing):\n")
            f.write(f"  Total masks: {self.results['before']['metrics']['total_masks']}\n")
            f.write(f"  Processing time: {self.results['before']['metrics']['total_processing_time']:.4f}s\n")
            f.write(f"  Avg boundary sharpness: {self.results['before']['metrics']['avg_boundary_sharpness']:.4f}\n\n")
            
            f.write("AFTER (Multi-View Refinement):\n")
            f.write(f"  Total masks: {self.results['after']['metrics']['total_masks']}\n")
            f.write(f"  Processing time: {self.results['after']['metrics']['total_processing_time']:.4f}s\n")
            f.write(f"  Avg boundary sharpness: {self.results['after']['metrics']['avg_boundary_sharpness']:.4f}\n\n")
            
            f.write("COMPARISON:\n")
            f.write(f"  Pixel change ratio: {comparison_metrics['pixel_change_ratio']:.4f}\n")
            f.write(f"  Speedup ratio: {comparison_metrics['speedup_ratio']:.2f}x\n")
            f.write(f"  Avg mask count change: {comparison_metrics['avg_mask_count_change']:.2f}\n")
        
        print(f"Benchmark report saved to: {report_file}")
        print(f"Summary saved to: {summary_file}")
        
        return report
    
    def run_benchmark(self, sample_views=10):
        """Run complete benchmark comparison"""
        print("Starting Multi-View SAM Mask Refinement Benchmark")
        print(f"Scene: {self.scene_path}")
        print(f"Sample views: {sample_views}")
        print("-" * 50)
        
        # Benchmark single-view processing
        before_masks, before_metrics = self.benchmark_single_view_processing(sample_views)
        
        if not before_masks:
            print("No SAM masks found in scene. Cannot run benchmark.")
            return None
        
        # Benchmark multi-view refinement
        after_masks, after_metrics = self.benchmark_multiview_refinement(before_masks, sample_views)
        
        # Compute comparison metrics
        comparison_metrics = self.compute_comparison_metrics(before_masks, after_masks)
        
        # Generate report
        report = self.generate_report(before_masks, after_masks, comparison_metrics)
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Views processed: {comparison_metrics['views_processed']}")
        print(f"Single-view time: {before_metrics['total_processing_time']:.4f}s")
        print(f"Multi-view time: {after_metrics['total_processing_time']:.4f}s")
        print(f"Pixel change ratio: {comparison_metrics['pixel_change_ratio']:.4f}")
        print(f"Boundary sharpness change: {after_metrics['avg_boundary_sharpness'] - before_metrics['avg_boundary_sharpness']:+.4f}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark multi-view SAM mask refinement")
    parser.add_argument('-s', '--source_path', required=True, help="Path to scene data")
    parser.add_argument('-o', '--output_dir', default='benchmark_results', help="Output directory for results")
    parser.add_argument('-n', '--num_views', type=int, default=10, help="Number of views to sample for benchmark")
    
    args = parser.parse_args()
    
    # Check if scene exists
    if not os.path.exists(args.source_path):
        print(f"Error: Scene path {args.source_path} does not exist")
        return
    
    # Run benchmark
    try:
        benchmark = MultiViewSAMBenchmark(args.source_path, args.output_dir)
        benchmark.run_benchmark(args.num_views)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()