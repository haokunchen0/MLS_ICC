export CUDA_VISIBLE_DEVICES=0

export MASTER_ADDR='localhost'
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export PYTHONPATH="$PYTHONPATH:open_flamingo"

LM_PATH= # llama model path 
LM_TOKENIZER_PATH= # llama model path
CKPT_PATH= # ckeckpoint path
VISION_ENCODER="ViT-L-14"
VISION_ENCODER_PRETRAINED='openai'
CACHED_PATH= # cache path

DATA_PATH= # data path
DATA_NAME= # data name

# DATA_NAME="cub200"

# DATA_NAME="stanford_dog"

# DATA_NAME="stanford_car"


RESULTS_FILE="results/results_LDE_$DATA_NAME.json"


python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --vision_encoder_path $VISION_ENCODER \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --dataset_name $DATA_NAME \
    --cached_demonstration_features $CACHED_PATH \
    --dataset_root $DATA_PATH  \
    --rices_type "image" \
    --cross_attn_every_n_layers 1 \
    --precision amp_bf16 \
    --shots 1 2 4 \
    --method_type "ML" \
    --num_samples -1 \
    --results_file $RESULTS_FILE \
    --Label_Distribution \
