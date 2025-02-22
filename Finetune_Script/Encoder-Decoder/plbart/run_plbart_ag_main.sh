CUDA_IDS=$1
OUTPUT_DIR=$2
DATA_PATH=$3

if [[ $DATA_PATH =~ ([Oo][lL][dD]) || $DATA_PATH =~([Nn][Ee][Ww]) ]]; then
    extractedString="${BASH_REMATCH[1]}"
    extractedStringLower=$(echo $extractedString | tr '[:upper:]' '[:lower:]')
    echo "DataSet Type is $extractedStringLower"
else
    echo "DataSet type not recognize!"
fi

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ../../encoder_decoder_ag_main.py \
            --do_train \
            --do_eval \
            --do_predict \
            --model_name_or_path uclanlp/plbart-base \
            --source_max_length 512 \
            --target_max_length 256 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 4 \
            --auto_find_batch_size \
            --dataloader_num_workers 4 \
            --dataloader_prefetch_factor 2 \
            --dataloader_pin_memory \
            --gradient_accumulation_steps 1 \
            --data_path ${DATA_PATH} \
            --train_filename assert_train_${extractedStringLower}.jsonl \
            --eval_filename assert_eval_${extractedStringLower}.jsonl \
            --test_filename assert_test_${extractedStringLower}.jsonl \
            --learning_rate 5e-5 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --greater_is_better False \
            --logging_steps 50 \
            --num_train_epochs 75 \
            --save_total_limit 3 \
            --output_dir ${OUTPUT_DIR} \
            --save_safetensors False \
            --plot_loss \
            --model_type plbart \
            --max_new_tokens 256 \
            --max_length 1024