#!/usr/bin/env bash

data_outdir="data_preprocessed"
model_outdir="model"
max_length=150
batch_size=4
num_epochs=5
learning_rate="2e-5"

usage="Usage:\n\t$0 [-i infile] [-m model_path] <-d data_outdir> <-o model_outdir> <-l max_length> <-b batch_size> <-n num_epochs> <-r learning_rate>"

path=$(dirname $(realpath "$BASH_SOURCE"))

function main {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                echo -e $usage
                return
                ;;
            -i)
                infile="$2"
                shift
                shift
                ;;
            -m)
                model_path="$2"
                shift
                shift
                ;;
            -d)
                data_outdir="$2"
                shift
                shift
                ;;
            -o)
                model_outdir="$2"
                shift
                shift
                ;;
            -l)
                max_length="$2"
                shift
                shift
                ;;
            -b)
                batch_size="$2"
                shift
                shift
                ;;
            -n)
                num_epochs="$2"
                shift
                shift
                ;;
            -r)
                learning_rate="$2"
                shift
                shift
                ;;
            *)
                echo "ERROR: Bad option '$1'." >&2
                echo -e $usage
                return -1
                ;;
        esac
    done
    
if [ ! -d $data_outdir ];then
    mkdir $data_outdir
fi

if [ ! -d $model_outdir ];then
    mkdir $model_outdir
fi

    "$path"/run_pipeline.py $infile $model_path $data_outdir $model_outdir $max_length $batch_size $num_epochs $learning_rate
    
    }

if ! { ( return ) } 2>/dev/null; then
    set -e
    main "$@" || exit
fi
