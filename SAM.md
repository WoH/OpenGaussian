# Multi-View SAM Mask Refinement

## Overview
Refine instance segmentation masks by enforcing consistency of SAM-based masks across overlapping views. This approach improves mask quality by leveraging multi-view geometric constraints while maintaining compatibility with the existing OpenGaussian pipeline.

## Current SAM Processing Pipeline
- **Location**: `utils/opengs_utlis.py:125-182` - `get_SAM_mask_and_feat()`
- **Input**: `gt_sam_mask[4, H, W]` - hierarchical SAM mask IDs
- **Output**: `mask_bool[num_mask, H, W]` - boolean masks for each instance
- **Usage**: `train.py:377-400` - single view processing during training

## Multi-View Refinement Architecture

### Step 1: View Overlap Detection
**Function**: `detect_view_overlaps(cameras, depth_maps) -> overlap_matrix`
- **Input**: Camera poses, intrinsics, rendered depth maps
- **Output**: Overlap matrix `[N_views, N_views]` with overlap percentages
- **Method**: Ray-based overlap computation using camera frustums and depth

### Step 2: Mask Reprojection
**Function**: `reproject_masks_between_views(mask_bool, cameras, depth_maps, view_pairs)`
- **Input**: Boolean masks from source view, camera parameters, depth maps
- **Output**: Reprojected masks in target views
- **Method**: 
  1. Use depth + camera matrices to compute 3D points
  2. Reproject 3D points to target view
  3. Handle occlusions using depth testing
  4. Interpolate mask boundaries

### Step 3: Consistency Measurement
**Function**: `measure_mask_consistency(original_masks, reprojected_masks) -> consistency_scores`
- **Metrics**:
  - IoU between original and reprojected masks
  - Boundary alignment using edge detection
  - Feature similarity in overlapping regions
- **Output**: Per-mask consistency scores `[num_masks]`

### Step 4: Mask Refinement Algorithm
**Function**: `refine_sam_masks_multiview(sam_masks_dict, cameras, depth_maps) -> refined_masks_dict`

**Algorithm**:
```
For each view:
  1. Start with single-view SAM masks
  2. Find overlapping views using overlap detection
  3. Reproject masks to/from overlapping views
  4. Measure consistency with reprojected masks
  5. Identify inconsistent mask regions
  6. Refine boundaries using weighted voting:
     - Higher weight for views with better visibility
     - Lower weight for occluded regions
  7. Update mask boundaries
  8. Repeat until convergence (max 3 iterations)
```

### Step 5: Integration Points

#### Modified Functions:
1. **`utils/opengs_utlis.py`**:
   - Extend `get_SAM_mask_and_feat()` to optionally use multi-view refinement
   - Add parameter `use_multiview_refinement=False` for backward compatibility

2. **`train.py:377-400`**:
   - Pass camera information and depth maps to SAM mask processing
   - Enable multi-view refinement during training

#### New Functions:
- `detect_view_overlaps()`
- `reproject_masks_between_views()`
- `measure_mask_consistency()`
- `refine_sam_masks_multiview()`

## Implementation Details

### Data Structures:
- **Input**: Maintain existing `gt_sam_mask[4, H, W]` format
- **Output**: Keep `mask_bool[num_mask, H, W]` boolean mask format
- **Internal**: Add `overlap_matrix`, `consistency_scores`, `refinement_weights`

### Key Considerations:
1. **Occlusion Handling**: Use depth testing to identify occluded regions
2. **Boundary Preservation**: Maintain sharp mask boundaries during refinement
3. **Computational Efficiency**: Only process overlapping view pairs
4. **Backward Compatibility**: Default to single-view processing if multi-view disabled

### Dependencies:
- Camera poses from `scene/cameras.py`
- Rendered depth maps from Gaussian splatting
- Existing SAM mask processing pipeline
- PyTorch geometric operations for reprojection

## Expected Benefits:
1. **Improved Mask Quality**: More accurate boundaries through multi-view consensus
2. **Reduced Noise**: Filtering of SAM artifacts using geometric constraints
3. **Better 3D Consistency**: Masks that are geometrically coherent across views
4. **Maintained Performance**: Backward compatible with existing training pipeline

## Testing Strategy:
1. **Unit Tests**: Individual function testing with synthetic data
2. **Integration Tests**: End-to-end pipeline with real SAM masks
3. **Ablation Study**: Compare single-view vs multi-view refined masks
4. **Quantitative Metrics**: IoU, boundary accuracy, 3D consistency measures