export CUDA_VISIBLE_DEVICES=3

export MASTER_ADDR='localhost'
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export PYTHONPATH="$PYTHONPATH:open_flamingo"

LM_PATH="/data/hyh/mpt-1b-redpajama-200b" # llama model path 
LM_TOKENIZER_PATH="/data/hyh/mpt-1b-redpajama-200b" # llama model path
CKPT_PATH="/data/hyh/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
VISION_ENCODER="ViT-L-14"
VISION_ENCODER_PRETRAINED='openai'
CACHED_PATH="/data/zihan/icl"

DATA_PATH="/data/hyh/imagenet/data/"
DATA_NAME="imagenet"

# DATA_NAME="cub200"
# DATA_PATH="/data/hyh/CUB_200_2011/"

# DATA_NAME="stanford_dog"
# DATA_PATH="/data/hyh/Dog/"

# DATA_NAME="stanford_car"
# DATA_PATH="/data/hyh/car_data/car_data/"

RESULTS_FILE_1="results/results_RS_$DATA_NAME.json"
RESULTS_FILE_2="results/results_SIIR_$DATA_NAME.json"
RESULTS_FILE_3="results/results_SILR_$DATA_NAME.json"


# echo "Start RS evaluation on dataset $DATA_NAME"
# python open_flamingo/eval/evaluate.py \
#     --lm_path $LM_PATH \
#     --lm_tokenizer_path $LM_TOKENIZER_PATH \
#     --checkpoint_path $CKPT_PATH \
#     --dataset_name $DATA_NAME \
#     --vision_encoder_path $VISION_ENCODER \
#     --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
#     --cross_attn_every_n_layers 1 \
#     --precision amp_bf16 \
#     --dataset_root $DATA_PATH  \
#     --num_samples -1 \
#     --results_file $RESULTS_FILE_1 \

echo "Start SIIR evaluation on dataset $DATA_NAME"
python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --dataset_name $DATA_NAME \
    --vision_encoder_path $VISION_ENCODER \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --cached_demonstration_features $CACHED_PATH \
    --rices_type "image" \
    --cross_attn_every_n_layers 1 \
    --precision amp_bf16 \
    --dataset_root $DATA_PATH  \
    --num_samples -1 \
    --results_file $RESULTS_FILE_2 \

# echo "Start SICR text evaluation on dataset $DATA_NAME"
# python open_flamingo/eval/evaluate.py \
#     --lm_path $LM_PATH \
#     --lm_tokenizer_path $LM_TOKENIZER_PATH \
#     --checkpoint_path $CKPT_PATH \
#     --dataset_name $DATA_NAME \
#     --vision_encoder_path $VISION_ENCODER \
#     --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
#     --cached_demonstration_features $CACHED_PATH \
#     --rices_type "text" \
#     --cross_attn_every_n_layers 1 \
#     --precision amp_bf16 \
#     --dataset_root $DATA_PATH  \
#     --num_samples -1 \
#     --results_file $RESULTS_FILE_3 \

echo "evaluation complete!"