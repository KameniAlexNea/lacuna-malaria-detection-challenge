nohup python train_hf.py \
    --output_dir logs \
    --run_name cond-detr-50 \
    --auto_find_batch_size \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 200 \
    --do_eval \
    --do_train \
    --fp16 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --save_total_limit 3 \
    --remove_unused_columns false \
    --push_to_hub false \
    --eval_strategy epoch \
    --save_strategy epoch \
    --report_to wandb \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --data_seed 41 \
    --save_safetensors \
    --save_only_model \
    --metric_for_best_model map_50 \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps 2 \
    &> nohup.out &