for seed in 42 43 44 45 46; do
    for task_name in opentable_binary; do  # opentable_ternary opentable_5_way; do
        for model_name in bert-base-uncased; do  # roberta-base t5-base gpt2 lstm; do
            instance_path="${task_name}/${model_name}/None__None/seed_${seed}"

            # training script for the no-confounding case
            python run_opentable.py \
                --seed ${seed} \
                --task_name ${task_name} \
                --model_name_or_path ${model_name} \
                --group run_opentable__${task_name}__${model_name}__None__None \
                --output_dir ./outputs_run_opentable/${instance_path} \
                \
                --do_train \
                --do_eval \
                \
                --max_seq_length 128 \
                --per_device_train_batch_size 64 \
                --learning_rate 2e-5 \
                --num_train_epochs 10 \
                \
                --save_total_limit 1 \
                --cache_dir ./train_cache/ \
                --report_to wandb \
                --logging_steps 10

            # run inference and save predictions
            python predictions.py \
                --model_name_or_path ./outputs_run_opentable/${instance_path} \
                --task_name ${task_name} \
                --model_architecture ${model_name} \
                --output_dir ./outputs_predictions/${instance_path}
        done
    done
done