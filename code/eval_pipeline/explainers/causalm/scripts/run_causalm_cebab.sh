#!/bin/bash

finetunning_per_device_train_batch_size=128;
finetunning_per_device_eval_batch_size=128;
finetunning_learning_rate=1e-3;
finetunning_num_train_epochs=50;

for seed in 44; do
    for task_name in opentable_ternary; do
        for model_architecture in gpt2; do
            rm -r ~/.cache/huggingface/datasets

            # factual training
#            python run_causalm_cebab.py \
#                --seed ${seed} \
#                --task_name ${task_name} \
#                --model_name_or_path ${model_architecture} \
#                --model_architecture ${model_architecture} \
#                --output_dir ./outputs_cebab/${task_name}/${model_architecture}/None__None/seed_${seed} \
#                \
#                --counterfactual_training False \
#                --do_train \
#                --do_eval \
#                --label_names task_labels \
#                --evaluation_strategy steps \
#                \
#                --max_seq_length 128 \
#                --per_device_train_batch_size ${finetunning_per_device_train_batch_size} \
#                --per_device_eval_batch_size ${finetunning_per_device_eval_batch_size} \
#                --learning_rate ${finetunning_learning_rate} \
#                --num_train_epochs ${finetunning_num_train_epochs} \
#                \
#                --save_total_limit 1 \
#                --report_to none \
#                --logging_steps 60 \
#                --save_strategy no \
#                --overwrite_cache True

            # counterfactual training
            for tc in service; do
                # control concept
                if [[ ${tc} == "food" ]]; then
                    cc=service;
                else
                    cc=food;
                fi

                # tc num labels
                if [[ ${tc} == "ambiance" ]] || [[ ${tc} == "food" ]] || [[ ${tc} == "noise" ]] || [[ ${tc} == "service" ]]; then
                    tc_heads_num_labels=3;
                else
                    tc_heads_num_labels=2;
                fi

                # cc num labels
                if [[ ${cc} == "ambiance" ]] || [[ ${cc} == "food" ]] || [[ ${cc} == "noise" ]] || [[ ${cc} == "service" ]]; then
                    cc_heads_num_labels=3;
                elif [[ ${cc} == "cuisine" ]]; then
                    cc_heads_num_labels=5;
                else
                    cc_heads_num_labels=2;
                fi

                instance_path="${task_name}/${model_architecture}/${tc}__${cc}/seed_${seed}"

                rm -r ~/.cache/huggingface/datasets

                python run_causalm_cebab.py \
                    --seed ${seed} \
                    --task_name ${task_name} \
                    --model_name_or_path ${model_architecture} \
                    --model_architecture ${model_architecture} \
                    --output_dir ./outputs_causalm/${instance_path} \
                    \
                    --causalm_additional_pretraining \
                    --tc_labels_col ${tc}_aspect_majority \
                    --cc_labels_col ${cc}_aspect_majority \
                    --num_tc 1 \
                    --num_cc 1 \
                    --tc_heads_types seq_classification \
                    --cc_heads_types seq_classification \
                    --tc_heads_num_labels ${tc_heads_num_labels} \
                    --cc_heads_num_labels ${cc_heads_num_labels} \
                    --tc_lambda 0.1 \
                    \
                    --max_seq_length 128 \
                    --per_device_train_batch_size 24 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    \
                    --save_total_limit 1 \
                    --cache_dir ./hf_cache \
                    --report_to none \
                    --logging_steps 10 \
                    --save_strategy no \
                    --overwrite_cache True

                rm -r ~/.cache/huggingface/datasets

                python run_causalm_cebab.py \
                    --seed ${seed} \
                    --task_name ${task_name} \
                    --model_name_or_path ./outputs_causalm/${instance_path}/additional_pretraining \
                    --model_architecture ${model_architecture} \
                    --output_dir ./outputs_causalm/${instance_path} \
                    \
                    --do_train \
                    --do_eval \
                    --label_names task_labels \
                    --evaluation_strategy steps \
                    \
                    --max_seq_length 128 \
                    --per_device_train_batch_size ${finetunning_per_device_train_batch_size} \
                    --per_device_eval_batch_size ${finetunning_per_device_eval_batch_size} \
                    --learning_rate ${finetunning_learning_rate} \
                    --num_train_epochs ${finetunning_num_train_epochs} \
                    \
                    --save_total_limit 1 \
                    --report_to none \
                    --logging_steps 60 \
                    --save_strategy no \
                    --overwrite_cache True
            done
        done
    done
done