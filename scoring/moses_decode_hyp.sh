#!/bin/bash

TOOLS=/mnt/disks/adisk/tools
SCRIPTS=$TOOLS/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BASE=/mnt/disks/adisk/true_nmt/scoring
for dir in mbart m2m100; do
    for src in en fr; do
        for tgt in en fr; do
            if [ "$src" != "$tgt" ]; then
                for f in test.hyp valid.hyp; do
                    f=$BASE/$dir/$src"_"$tgt/$f
                    # echo $(wc -l $f)
                    rm $f.decoded
                    cat $f | \
                        perl $NORM_PUNC $l | \
                        perl $REM_NON_PRINT_CHAR | \
                        perl $TOKENIZER -threads 8 -a -l $tgt > $f.decoded
                done
            fi
        done
    done
done