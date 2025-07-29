#!/bin/bash
# Complete pipeline for multi-view CLIP feature fusion in OpenGaussian

# Configuration
MODEL_PATH="output/your_model_path"  # Update this
SCENE_PATH="data/scannet/scene0000_00"  # Update this
FEATURE_DIR="${SCENE_PATH}/language_features"
VISIBILITY_DIR="${MODEL_PATH}/visibility"
FUSED_FEATURES_DIR="${MODEL_PATH}/fused_features"

# Create output directories
mkdir -p ${VISIBILITY_DIR}
mkdir -p ${FUSED_FEATURES_DIR}

echo "=== Multi-View CLIP Feature Fusion Pipeline ==="
echo "Model path: ${MODEL_PATH}"
echo "Scene path: ${SCENE_PATH}"
echo ""

# Step 1: Compute visibility mapping
echo "Step 1: Computing Gaussian visibility across all views..."
python scripts/compute_visibility.py \
    --model_path ${MODEL_PATH} \
    --source_path ${SCENE_PATH} \
    --output_dir ${VISIBILITY_DIR} \
    --margin 1.2

# Check if visibility files were created
if [ ! -f "${VISIBILITY_DIR}/visibility_train.json" ]; then
    echo "Error: Visibility computation failed"
    exit 1
fi

echo ""
echo "Step 2: Extracting and fusing CLIP features..."

# Step 2a: Extract fused features with uniform weighting
echo "  2a: Uniform weighting..."
python scripts/extract_fused_features.py \
    --model_path ${MODEL_PATH} \
    --source_path ${SCENE_PATH} \
    --visibility_json ${VISIBILITY_DIR}/visibility_train.json \
    --feature_dir ${FEATURE_DIR} \
    --output_path ${FUSED_FEATURES_DIR}/fused_features_train_uniform.npy \
    --split train

# Step 2b: Extract fused features with distance weighting
echo "  2b: Distance-based weighting..."
python scripts/extract_fused_features.py \
    --model_path ${MODEL_PATH} \
    --source_path ${SCENE_PATH} \
    --visibility_json ${VISIBILITY_DIR}/visibility_train.json \
    --feature_dir ${FEATURE_DIR} \
    --output_path ${FUSED_FEATURES_DIR}/fused_features_train_distance.npy \
    --use_visibility_weights \
    --weight_type distance \
    --split train

# Step 2c: Extract fused features with angle weighting
echo "  2c: Angle-based weighting..."
python scripts/extract_fused_features.py \
    --model_path ${MODEL_PATH} \
    --source_path ${SCENE_PATH} \
    --visibility_json ${VISIBILITY_DIR}/visibility_train.json \
    --feature_dir ${FEATURE_DIR} \
    --output_path ${FUSED_FEATURES_DIR}/fused_features_train_angle.npy \
    --use_visibility_weights \
    --weight_type angle \
    --split train

echo ""
echo "Step 3: Training with fused features..."
echo "You can now train with fused features using:"
echo ""
echo "python train.py \\"
echo "    -s ${SCENE_PATH} \\"
echo "    -m ${MODEL_PATH}_fused \\"
echo "    --use_fused_features \\"
echo "    --fused_features_path ${FUSED_FEATURES_DIR}/fused_features_train_distance.npy \\"
echo "    --visibility_json_path ${VISIBILITY_DIR}/visibility_train.json \\"
echo "    --use_visibility_weights \\"
echo "    --visibility_weight_type distance \\"
echo "    [other training arguments]"
echo ""
echo "=== Pipeline Complete ===">