#!/usr/bin/env bash
# run_train.sh - 使用本地 BioCLIP2 作为教师模型（ViT-L/14）

export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# 路径配置
TRAIN_DATA="/home/111_wcy/work/CLIP-KD/datasets/data_csv/train.csv"
VAL_DATA="/home/111_wcy/work/CLIP-KD/datasets/data_csv/val.csv"
LOG_ROOT="./logs"

# 学生模型和教师模型
STUDENT_MODEL="ViT-B-32"
TEACHER_MODEL="ViT-L-14"   # BioCLIP2 基于 ViT-L/14 框架
TEACHER_CKPT="/home/111_wcy/work/CLIP-KD/models/bioclip-2/open_clip_pytorch_model.bin"

# 超参数
BATCH_SIZE=4
LR=1e-4
EPOCHS=10
WARMUP=100
WORKERS=1

python src/training/main_me.py \
  --model "${STUDENT_MODEL}" \
  --t-model "${TEACHER_MODEL}" \
  --t-model-checkpoint "${TEACHER_CKPT}" \
  --train-data "${TRAIN_DATA}" \
  --val-data "${VAL_DATA}" \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --epochs ${EPOCHS} \
  --warmup ${WARMUP} \
  --precision amp \
  --workers ${WORKERS} \
  --logs "${LOG_ROOT}" \
  --report-to none