#!/bin/bash

set -e

# if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
#     gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
# else
#     IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
#     gpu_count=${#devices[@]}
# fi

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=2
port_num=29500
while netstat -tuln | grep -q ":$port_num"; do
    ((port_num++))
done
num_frames=32
launcher=accelerate

export OPENAI_API_KEY="" # API KEY FOR OPENAI CHATGPT
export GOOGLE_API_KEY="" # API KEY FOR GOGOLE GEMINI

export HF_ENDPOINT=https://hf-mirror.com

available_models="internvl3_8b_8f_spa"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "internvl3_1b_8f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-1B,modality=video,max_frames_num=8"
        ;;
    "internvl3_1b_8f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-1B,modality=video,max_frames_num=8,use_spatial=True"
        ;;
    "internvl3_2b_8f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-2B,modality=video,max_frames_num=8"
        ;;
    "internvl3_2b_8f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-2B,modality=video,max_frames_num=8,use_spatial=True"
        ;;
    "internvl3_8b_8f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=8"
        ;;
    "internvl3_8b_12f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=12"
        ;;
    "internvl3_8b_16f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=16"
        ;;
    "internvl3_8b_24f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=24"
        ;;
    "internvl3_8b_36f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=36"
        ;;
    "internvl3_8b_8f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=8,use_spatial=True"
        ;;
    "internvl3_8b_12f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=12,use_spatial=True"
        ;;
    "internvl3_8b_16f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=16,use_spatial=True"
        ;;
    "internvl3_8b_24f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=24,use_spatial=True"
        ;;
    "internvl3_8b_32f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=32,use_spatial=True"
        ;;
    "internvl3_14b_8f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-14B,modality=video,max_frames_num=8"
        ;;
    "internvl3_14b_8f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-14B,modality=video,max_frames_num=8,use_spatial=True"
        ;;
    "internvl3_14b_32f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-14B,modality=video,max_frames_num=32"
        ;;
    "internvl3_14b_32f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-14B,modality=video,max_frames_num=32,use_spatial=True"
        ;;
    "internvl3_8b_32f_spa")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=32,use_spatial=True"
        ;;
    "internvl3_8b_32f")
        model_family="internvl2"
        model_args="pretrained=checkpoint-local/InternVL3-8B,modality=video,max_frames_num=32"
        ;;
    *)

        echo "Unknown model: $model"
        exit -1
        ;;
    esac

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            --main_process_port=$port_num \
            "
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path/$benchmark \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done
