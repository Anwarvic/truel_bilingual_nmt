#!/bin/bash

# changes these variables
MODEL_NAME=bi+csw
DATA_NAME=csw
SRC=src
TGT=tgt
GROUP=valid

# don't change these variables
BASE=/mnt/disks/adisk
MODEL_DICT=$BASE/data/fr_en/$MODEL_NAME/data-bin
DATA=$BASE/data/fr_en/$DATA_NAME
MODEL=$BASE/true_nmt/checkpoints/transformer_base/$MODEL_NAME/checkpoint_best.pt
OUT=$BASE/true_nmt/scoring/$MODEL_NAME/$DATA_NAME
mkdir -p $OUT

# activate virtual environment
source $BASE/true_nmt/py38/bin/activate

# disable warnings
export PYTHONWARNINGS="ignore"

# rm $OUT/$GROUP.txt
# nohup fairseq-interactive $MODEL_DICT \
#     --input $DATA/$GROUP.$SRC \
#     --path $MODEL \
#     -s $SRC -t $TGT \
#     --num-workers 5 \
#     --results-path $OUT \
#     --gen-subset $GROUP \
#     --max-tokens 4096 \
#     --beam 5 \
#     --sacrebleu \
#     --remove-bpe \
#     >> $OUT/$GROUP.txt 2>&1 &
# wait

echo "Started scoring..."
cat $DATA/$GROUP.$TGT | awk '{gsub("@@ ",""); print "T-"NR-1"	"$0;}' >> $OUT/$GROUP.txt
grep ^S $OUT/$GROUP.txt | cut -f2- > $OUT/$GROUP.src
grep ^H $OUT/$GROUP.txt | cut -f3- > $OUT/$GROUP.hyp
grep ^T $OUT/$GROUP.txt | cut -f2- > $OUT/$GROUP.ref

echo "Splitting files into English & French"
rm $OUT/$GROUP-fr.hyp $OUT/$GROUP-fr.ref
rm $OUT/$GROUP-en.hyp $OUT/$GROUP-en.ref
while read l1 <&3 && read l2 <&4 && read l3 <&5; do
    if [[ $l1 == "<2en>"* ]]; then
        echo $l2 >> $OUT/$GROUP-en.hyp
        echo $l3 >> $OUT/$GROUP-en.ref
    else
        echo $l2 >> $OUT/$GROUP-fr.hyp
        echo $l3 >> $OUT/$GROUP-fr.ref
    fi
done 3<$OUT/$GROUP.src 4<$OUT/$GROUP.hyp 5<$OUT/$GROUP.ref

echo "Computing the BLEU score"
for l in en fr; do
    echo -e "\tScoring $l..."
    fairseq-score --sys $OUT/$GROUP-$l.hyp --ref $OUT/$GROUP-$l.ref
done