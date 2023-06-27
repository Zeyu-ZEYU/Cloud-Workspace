#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=./pretrain_gpt/checkpoints
VOCAB_FILE=./pretrain_gpt/vocab.json
MERGE_FILE=./pretrain_gpt/merges.txt
DATA_PATH=./pretrain_gpt/data/BookCorpusDataset_text_document

GPT_ARGS="
    --num-layers 12 \
    --tensor-model-parallel-size 8 \
    --hidden-size 512 \
    --num-attention-heads 8 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr="172.31.92.228" \
         --master_port=34728 \
         pretrain_gpt.py \
         $GPT_ARGS \
         $DATA_ARGS \
         $OUTPUT_ARGS \
         --save $CHECKPOINT_PATH \
         --load $CHECKPOINT_PATH
