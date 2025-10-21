#!/bin/bash

# Script to run CSReL (Coreset Selection for Continual Learning) with mammoth framework on CIFAR datasets
# Based on the original CSReL-Coreset-CL implementation and adapted for mammoth framework
# 
# Usage: ./run_csrel_cifar.sh [dataset] [buffer_size] [lr] [ref_epochs] [ref_lr] [device]
# Example: ./run_csrel_cifar.sh seq-cifar10 500 0.1 10 0.01 cuda:0

set -e  # Exit on any error

echo "====================================================="
echo "CSReL Continual Learning with Mammoth Framework"
echo "====================================================="

# Set default parameters
DATASET=${1:-"seq-cifar10"}
BUFFER_SIZE=${2:-200}
LR=${3:-0.001}
REF_EPOCHS=${4:-150}
REF_LR=${5:-0.003}
DEVICE=${6:-"cuda:0"}
SEED=${7:-0}
EPOCHS=${8:-400}
BATCH_SIZE=${9:-256}
MINIBATCH_SIZE=${10:-32}

# CSReL-specific parameters
CSREL_CE_FACTOR=${11:-1.0}
CSREL_MSE_FACTOR=${12:-0.0}
CSREL_BATCH_SIZE=${13:-32}
CSREL_SELECTION_STEPS=${14:-1}
CSREL_CLASS_BALANCE=${15:-"true"}

# Additional parameters
BACKBONE=${16:-"resnet18"}
VALIDATION=${17:-0}
SAVECHECK=${18:-"false"}
NOWAND=${19:-1}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Buffer size: $BUFFER_SIZE"
echo "  Learning rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Minibatch size: $MINIBATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Seed: $SEED"
echo ""
echo "CSReL Parameters:"
echo "  Reference epochs: $REF_EPOCHS"
echo "  Reference LR: $REF_LR"
echo "  CE factor: $CSREL_CE_FACTOR"
echo "  MSE factor: $CSREL_MSE_FACTOR"
echo "  CSReL batch size: $CSREL_BATCH_SIZE"
echo "  Selection steps: $CSREL_SELECTION_STEPS"
echo "  Class balance: $CSREL_CLASS_BALANCE"
echo "  Backbone: $BACKBONE"
echo "====================================================="


# Validate device
if [[ $DEVICE == cuda* ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️  CUDA device specified but nvidia-smi not found. Falling back to CPU."
        DEVICE="cpu"
    else
        echo "✅ CUDA device available: $DEVICE"
    fi
elif [[ $DEVICE == "cpu" ]]; then
    echo "✅ Using CPU device"
else
    echo "❌ Invalid device: $DEVICE. Use 'cpu' or 'cuda:0', 'cuda:1', etc."
    exit 1
fi

# Set environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directory
RESULTS_DIR="./results/csrel_${DATASET}_${BUFFER_SIZE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo "====================================================="

# Build the command
CMD="python3.11 main.py \
    --dataset $DATASET \
    --model csrel \
    --buffer_size $BUFFER_SIZE \
    --minibatch_size $MINIBATCH_SIZE \
    --lr $LR \
    --n_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --seed $SEED \
    --backbone $BACKBONE \
    --validation $VALIDATION \
    --csrel_ref_epochs $REF_EPOCHS \
    --csrel_ref_lr $REF_LR \
    --csrel_ce_factor $CSREL_CE_FACTOR \
    --csrel_mse_factor $CSREL_MSE_FACTOR \
    --csrel_batch_size $CSREL_BATCH_SIZE \
    --csrel_selection_steps $CSREL_SELECTION_STEPS"

# Add class balance flag if enabled
if [[ $CSREL_CLASS_BALANCE == "true" ]]; then
    CMD="$CMD --csrel_class_balance"
fi

# Add CSReL buffer settings
CMD="$CMD --csrel_use_coreset_buffer \
    --csrel_buffer_path ./csrel_buffer_${DATASET}_${BUFFER_SIZE}"

echo "Running command:"
echo "$CMD"
echo "====================================================="

# Run the command
eval $CMD

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "====================================================="
    echo "✅ CSReL training completed successfully!"
    echo "Results saved to: $RESULTS_DIR"
    echo "====================================================="
else
    echo "====================================================="
    echo "❌ CSReL training failed!"
    echo "Check the error messages above for details."
    echo "====================================================="
    exit 1
fi

echo ""
echo "💡 Tips for troubleshooting:"
echo "  - If CUDA out of memory, try reducing --batch_size or --csrel_batch_size"
echo "  - If training is slow, try reducing --csrel_selection_steps to 1"
echo "  - For faster training, use --csrel_ref_epochs 5"
echo "  - To use CPU, set device to 'cpu'"
echo ""
echo "📊 To monitor training progress:"
echo "  - Check the logs in the results directory"
echo "  - Use tensorboard if enabled"
echo "  - Monitor GPU usage with 'nvidia-smi'"


