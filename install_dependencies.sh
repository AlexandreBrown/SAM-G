#!/bin/bash
pip install -r ./baselines/samg/requirements.txt
pip uninstall efficientvit -y
pip install ./baselines/samg/efficientvit/

# Use the WEIGHTS_FOLDER environment variable if set, otherwise default to "weights"
: "${WEIGHTS_FOLDER:=weights}"

if [ ! -d "$WEIGHTS_FOLDER" ]; then
    mkdir -p "$WEIGHTS_FOLDER"
    echo "Created folder: $WEIGHTS_FOLDER"
fi

# Download EfficientViT-SAM weights
SAM_WEIGHTS_FILE="efficientvit_sam_l1.pt"
SAM_WEIGHTS_FILE_URL="https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/$SAM_WEIGHTS_FILE"
if [ ! -f "$WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE" ]; then
    echo "Downloading $SAM_WEIGHTS_FILE..."
    wget -q -O "$WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE" "$SAM_WEIGHTS_FILE_URL"
    echo "Download complete: $WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE"
else
    echo "EfficientViT-SAM weights already exists: $WEIGHTS_FOLDER/$SAM_WEIGHTS_FILE"
fi

########################################
# DINOv2 ViT-B/14 weights
########################################
DINOV2_FILE="dinov2_vitb14_pretrain.pth"
DINOV2_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/$DINOV2_FILE"

if [ ! -f "$WEIGHTS_FOLDER/checkpoints/$DINOV2_FILE" ]; then
    echo "Creating $WEIGHTS_FOLDER/checkpoints/"
    mkdir -p "$WEIGHTS_FOLDER/checkpoints/"
    echo "Downloading $DINOV2_FILE..."
    wget -q -O "$WEIGHTS_FOLDER/checkpoints/$DINOV2_FILE" "$DINOV2_URL"
    echo "Download complete: $WEIGHTS_FOLDER/checkpoints/$DINOV2_FILE"
else
    echo "DINOv2 weights already exist: $WEIGHTS_FOLDER/checkpoints/$DINOV2_FILE"
fi


######################################
# Resnet
######################################
RESNET_FILE="resnet18.pth"
RESNET_URL="https://download.pytorch.org/models/resnet18-f37072fd.pth"

if [ ! -f "$WEIGHTS_FOLDER/checkpoints/$RESNET_FILE" ]; then
    echo "Creating $WEIGHTS_FOLDER/checkpoints/"
    mkdir -p "$WEIGHTS_FOLDER/checkpoints/"
    echo "Downloading $RESNET_FILE..."
    wget -q -O "$WEIGHTS_FOLDER/checkpoints/$RESNET_FILE" "$RESNET_URL"
    echo "Download complete: $WEIGHTS_FOLDER/checkpoints/$RESNET_FILE"
else
    echo "ResNet-18 weights already exist: $WEIGHTS_FOLDER/checkpoints/$RESNET_FILE"
fi