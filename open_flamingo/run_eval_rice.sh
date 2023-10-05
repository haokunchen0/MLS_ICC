RESULTS_FILE="results/results_RICE_SL.json"

export MASTER_ADDR='localhost'
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export PYTHONPATH="$PYTHONPATH:open_flamingo"

LM_PATH="/home/chenhaokun/data/mpt-1b-redpajama-200b" # llama model path 
LM_TOKENIZER_PATH="/home/chenhaokun/data/mpt-1b-redpajama-200b" # llama model path
CKPT_PATH="/home/chenhaokun/data/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
VISION_ENCODER="ViT-L-14"
VISION_ENCODER_PRETRAINED='openai'
CACHED_PATH="/home/chenhaokun/data/icl"
DATA_PATH="/home/chenhaokun/data/imagenet/data/"
DATA_NAME="imagenet"

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --rices \
    --vision_encoder_path $VISION_ENCODER \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --cached_demonstration_features $CACHED_PATH \
    --results_file $RESULTS_FILE \
    --cross_attn_every_n_layers 1 \
    --precision amp_bf16 \
    --dataset_name $DATA_NAME \
    --dataset_root $DATA_PATH  \
    --no_caching_for_classification \
    --num_samples 5000 \
    --pkl_name "imagenet.pkl"

echo "evaluation complete! results written to $RESULTS_FILE"
