#!/bin/bash

BASE=/mnt/disks/adisk
SOURCE=$BASE/true_nmt
CHECKPOINT_DIR=$SOURCE/checkpoints/transformer_base/bi+csw
LOGDIR=$CHECKPOINT_DIR/logs
DATA=$BASE/data/fr_en/bi+csw/data-bin

mkdir -p $CHECKPOINT_DIR $LOGDIR

# activate virtual environment
source $SOURCE/py38/bin/activate

# disable warnings
export PYTHONWARNINGS="ignore"

# start training
nohup fairseq-train $DATA \
    --arch transformer \
    --save-dir $CHECKPOINT_DIR \
    --tensorboard-logdir $LOGDIR \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --num-workers 0 \
    --validate-interval-updates 6000 \
    --task translation \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch	50 \
    --patience 20 \
    >> $LOGDIR/training.log 2>&1 &
    # --seed 1 \
    # --max-tokens-valid 4096 \
