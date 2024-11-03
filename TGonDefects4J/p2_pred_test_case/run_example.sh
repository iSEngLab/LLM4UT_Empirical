CUDA_IDS=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_FILE=$3

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python pred_decoder_tg.py \
    --data_url none \
    --test_filename ./p1_parse_project/renew_merged.jsonl \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir ./save_result_decoder \
    --output_file $OUTPUT_FILE \
    --max_length 2046 \
    --max_new_tokens 512 \
    # --add_pad False \
    # --n_gpu 4 \
