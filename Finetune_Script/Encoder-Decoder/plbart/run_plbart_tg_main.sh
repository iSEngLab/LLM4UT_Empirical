CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3


PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ../../encoder_decoder_tg_main.py \
            --do_train \
            --do_eval \
            --do_predict \
            --model_name_or_path uclanlp/plbart-base \
            --source_max_length 1024 \
            --target_max_length 512 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 4 \
            --dataloader_num_workers 4 \
            --dataloader_pin_memory \
            --datalodaer_prefetch_factor 2 \
            --data_path ${DATA_PATH} \
            --train_filename test.jsonl \
            --eval_filename eval.jsonl \
            --test_filename train.jsonl \
            --learning_rate 5e-5 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --save_safetensors False \
            --greater_is_better False \
            --logging_steps 50 \
            --num_train_epochs 30 \
            --save_total_limit 3 \
            --output_dir ${OUTPUT_DIR} \
            --plot_loss \
            --max_new_tokens 512 \
            --max_length 1024 \
