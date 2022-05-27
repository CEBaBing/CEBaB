for seed in 42 43 44 45 46; do
    for task_name in opentable_binary; do  # opentable_ternary opentable_5_way; do
        for model_name in bert-base-uncased; do  # roberta-base t5-base gpt2 lstm; do
            for tc in ambiance food noise service; do
                instance_path="${task_name}/${model_name}/${tc}__None/seed_${seed}"
                # number of labels changes depending on the confounder
    #            if [ ${cc} == "cuisine" ]; then
    #                cc_heads_num_labels=5;
    #            elif [ ${cc} == "region" ]; then
    #                cc_heads_num_labels=4;
    #            else
    #                cc_heads_num_labels=3;  # positive unknown negative
    #            fi

                # run additional pretraining with adversarial heads
                python causalm.py \
                    --seed ${seed} \
                    --task_name ${task_name} \
                    --model_name_or_path ${model_name} \
                    --group causalm__${task_name}__${model_name}__${tc}__None \
                    --output_dir ./outputs_causalm/${instance_path} \
                    \
                    --causalm_additional_pretraining \
                    --tc_labels_col ${tc}_aspect_majority \
                    --num_tc 1 \
                    --tc_heads_types seq_classification \
                    --tc_heads_num_labels 3 \
                    --tc_lambda 0.2 \
                    \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 10 \
                    \
                    --save_total_limit 1 \
                    --cache_dir ./train_cache/ \
                    --report_to wandb \
                    --logging_steps 10 \
                    --save_strategy no

                # run downstream task training
                python causalm.py \
                    --seed ${seed} \
                    --task_name ${task_name} \
                    --model_name_or_path ./outputs_causalm/${instance_path}/additional_pretraining \
                    --group causalm__${task_name}__${model_name}__${tc}__None \
                    --output_dir ./outputs_causalm/${instance_path} \
                    \
                    --do_train \
                    \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 32 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 10 \
                    \
                    --save_total_limit 1 \
                    --cache_dir ./train_cache/ \
                    --report_to wandb \
                    --logging_steps 10 \
                    --save_strategy no

                # run inference and save predictions
                python predictions.py \
                    --is_causalm True \
                    --model_name_or_path ./outputs_causalm/${instance_path}/downstream \
                    --task_name ${task_name} \
                    --model_architecture ${model_name} \
                    --output_dir ./outputs_predictions/${task_name}/${model_name}/causalm__${tc}__None/seed_${seed}
            done
        done
    done
done
