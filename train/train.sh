export Teacher_PATH='meta-llama/Llama-3.2-3B-Instruct'
export Student_PATH='meta-llama/Llama-3.2-1B-Instruct'
export SAVE_PATH=$1
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

deepspeed --num_gpus=4 train.py \
    --teacher_model_name_or_path $Teacher_PATH \
    --student_model_name_or_path $Student_PATH \
    --dataset_name "eg-balanced" \
    --model_max_length 32 \
    --output_dir $SAVE_PATH \
    --logging_dir $2 \
    --num_train_epochs $3 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 15 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --train_kd True \
    --kd_loss_type "forward" \
    --max_train_samples 1000