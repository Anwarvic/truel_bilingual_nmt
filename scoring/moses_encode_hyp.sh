#!/bin/bash

TOOLS=/mnt/disks/adisk/tools
SCRIPTS=$TOOLS/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

BASE=/mnt/disks/adisk/true_nmt/scoring
# multilingual models like mbart & m2m100
for dir in mbart m2m100; do
    for folder in bidirectional csw; do
        for group in valid test; do
            for lang in en fr; do
                f=$BASE/$dir/$folder/$group"-"$lang.hyp
                # echo $(wc -l $f)
                rm $f.decoded
                cat $f | \
                    perl $NORM_PUNC $l | \
                    perl $REM_NON_PRINT_CHAR | \
                    perl $TOKENIZER -threads 8 -a -l $lang > $f.decoded
            done
        done
    done
done