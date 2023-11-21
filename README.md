# Manipulating the Label Space for In-Context Classification
[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/y2l/meta-transfer-learning-tensorflow/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

This repository includes the PyTorch implementation for our CVPR 2024 paper. 

#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
    - [Evaluation of In-Context Classification](#evaluation-of-in-context-classification)
    - [Generation of Visual Description](#generation-of-visual-description)
* [Datasets](#datasets)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)


## Introduction
After pre-training by generating the next word conditional on previous words, the Language Model (LM) acquires the ability of In-Context Learning (ICL) that can learn a new task conditional on the context of the given in-context examples (ICEs). Similarly, visually-conditioned Language Modelling is also used to train Vision-Language Models (VLMs) with ICL ability. However, such VLMs typically exhibit weaker classification abilities compared to contrastive learning-based models like CLIP, since the Language Modelling objective does not directly contrast whether an object is paired with a text. To improve the ICL of classification, using more ICEs to provide more knowledge is a straightforward way. However, this may largely increase the selection time, and more importantly, the inclusion of additional in-context images tends to extend the length of the in-context sequence beyond the processing capacity of a VLM. To alleviate these limitations, we propose to manipulate the label space of each ICE to increase its knowledge density, allowing for fewer ICEs to convey as much information as a larger set would. Specifically, we propose two strategies which are Label Distribution Enhancement and Visual Descriptions Enhancement to improve In-context classification performance on diverse datasets, including the classic ImageNet and more fine-grained datasets like CUB-200. Specifically, using our approach on ImageNet, we increase accuracy from $74.70\%$ in a $4$-shot setting to $76.21\%$ with just 2 shots. surpassing CLIP by $0.67\%$. On CUB-200, our method raises $1$-shot accuracy from $48.86\%$ to $69.05\%$, $12.15\%$ higher than CLIP.


![My SVG Image](assets/framework.svg)


> Figure:  Overview of our proposed Label Space Manipulating, in Label Distribution Enhancement (a), we use image features and text features extracted by CLIP for similarity calculation to get the top-$k$ similar images for a richer label context; In Visual Description Enhancement (b), we query the same VLM with targeted labels and their corresponding most relevant images to generate visual descriptions to assist classification. We explore both textual and visual-focused strategies while the whole VLM parameters are frozen.

## Getting Started

Create a conda environment for running the scripts, run
```bash
conda create -n openflamingo python=3.9
pip install -r requirements.txt
pip install -e .
```

Download the OpenFlamingo v2 3B model from this [link](https://huggingface.co/openflamingo/OpenFlamingo-3B-vitl-mpt1b).

Before starting the evaluation, make sure to cache both the image features (please refer to `cache_rices_image_features.py`) and textual features (please refer to `cache_rices_text_features.py`).

Use command:
```bash
python open_flamingo/eval/cache_rices_xxx_features.py --dataset_root xxx --output_dir xxx --batch_size xxx 
```
### Evaluation of In-Context Classification
Then, you can run the following command to evaluate the performance of in-context classification using OpenFlamingo. See `run_eval.sh` for more details.

```bash
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR='localhost'
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export PYTHONPATH="$PYTHONPATH:open_flamingo"

LM_PATH="" # llama model path 
LM_TOKENIZER_PATH="" # llama model path
CKPT_PATH=""
VISION_ENCODER="ViT-L-14"
VISION_ENCODER_PRETRAINED='openai'
CACHED_PATH=""

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

    # Optional parameters: 
    # --method_type is the label enhancement method in our paper.
    # --description is the visual enhancement method in our paper.
    # --ensemble is a combination of the two.
    # --OP is the LDE(DL) method in the thesis.
    # --Label_Distribution comes with --method_type "ML".
    # --rices_type is the way to retrieve ICE, add it and it will use RICES method.
    # --cached_demonstration_features is pre-computed features of image or text.
    # --shots, shots of ICE examples.
    # The above running code is using the LDE(DD) method under the RICES method
```
### Generation of Visual Description
If you want to generate visual description, please use the following command and change the prompt for discription in 'openflamingo/eval/models/open_flamingo.py' for your own custom discriptive prompt.

```bash
DATA_NAME=""
DATA_PATH=""
RESULTS_FILE="results/{$DATA_NAME}_description.txt"

python open_flamingo/eval/description.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --checkpoint_path $CKPT_PATH \
    --dataset_name $DATA_NAME \
    --vision_encoder_path $VISION_ENCODER \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --precision amp_bf16 \
    --dataset_root $DATA_PATH  \
    --num_samples -1 \
    --results_file $RESULTS_FILE \
    --batch_size 20 \
    --method_type "ETD" \
    --cross_attn_every_n_layers 1 \
    --cached_demonstration_features $CACHED_PATH \
```


## Datasets
[ImageNet](https://www.image-net.org/download.php),  [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/), [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) and  [Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) are chosen for evaluation.


## Citation

Please cite our paper if it is helpful to your work:


## TODO
Working on it...

## Acknowledgements

Our implementations is built upon the source code from [OpenFlamingo](https://github.com/mlfoundations/open_flamingo/tree/main).
