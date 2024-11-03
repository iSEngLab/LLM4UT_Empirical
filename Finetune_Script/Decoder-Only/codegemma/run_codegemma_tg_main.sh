CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3


# PYTHONUNBUFFERED=1  deepspeed --master_port 29501 --include localhost:${CUDA_IDS} ../decoder_tg_pred.py \
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ../../decoder_ag_main.py \
            --do_train \
            --do_eval \
            --do_predict \
            --model_name_or_path google/codegemma-2b \
            --model_max_length 2046 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 2 \
            --dataloader_num_workers 8 \
            --dataloader_pin_memory \
            --datalodaer_prefetch_factor 2 \
            --data_path ${DATA_PATH} \
            --train_filename train.jsonl \
            --eval_filename eval.jsonl \
            --test_filename test.jsonl \
            --learning_rate 5e-5 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --greater_is_better False \
            --logging_steps 50 \
            --num_train_epochs 50 \
            --save_total_limit 3 \
            --output_dir ${OUTPUT_DIR} \
            --plot_loss \
            --max_new_tokens 512 \
            --max_length 2046 \
            --add_pad True \
            # --use_deepspeed True \
