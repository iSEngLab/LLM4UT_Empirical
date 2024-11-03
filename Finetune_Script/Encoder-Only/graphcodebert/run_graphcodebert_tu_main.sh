
CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ../encoder_tu_main.py \
            --do_train \
            --do_eval \
            --do_predict \
            --model_name_or_path microsoft/graphcodebert-base \
            --model_type graphcodebert \
            --source_max_length 512 \
            --target_max_length 256 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --dataloader_num_workers 8 \
            --dataloader_pin_memory \
            --datalodaer_prefetch_factor 2 \
            --gradient_accumulation_steps 1 \
            --data_path ${DATA_PATH} \
            --train_filename filter_train.json \
            --eval_filename filter_valid.json \
            --test_filename filter_test.json \
            --learning_rate 5e-5 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --greater_is_better False \
            --logging_steps 50 \
            --num_train_epochs 50 \
            --save_total_limit 3 \
            --output_dir ${OUTPUT_DIR} \
            --plot_loss \
            --max_new_tokens 256 \
            --max_length 512