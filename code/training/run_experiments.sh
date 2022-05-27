# T5 Experiments - ABSA
# for condition in exclusive inclusive; do
#     for task_name in opentable-absa; do
#         for seed in 42 66 77 88 99; do
#             for model_name in t5-base; do
#                 # training script for the no-confounding case
#                 CUDA_VISIBLE_DEVICES=1,4,5,7 python run_cebab_t5.py \
#                 --model_name_or_path ${model_name} \
#                 --task_name ${task_name} \
#                 --dataset_name CEBaB/CEBaB.absa.${condition} \
#                 --do_train \
#                 --per_device_train_batch_size 32 \
#                 --max_source_length 128 \
#                 --learning_rate 2e-5 \
#                 --output_dir ./t5_cebab_absa_${condition}_results/ \
#                 --save_total_limit 1 \
#                 --cache_dir ../huggingface_cache/ \
#                 --seed ${seed} \
#                 --report_to none \
#                 --logging_steps 10
#             done
#         done
#     done
# done

# T5 Experiments - SA
# for condition in exclusive inclusive; do
#     for task_name in opentable; do
#         for seed in 42 66 77 88 99; do
#             for num_of_class in 2 3 5; do
#                 for model_name in t5-base; do
#                     # training script for the no-confounding case
#                     CUDA_VISIBLE_DEVICES=0,5,6,9 python run_cebab_t5.py \
#                     --model_name_or_path ${model_name} \
#                     --task_name ${task_name} \
#                     --dataset_name CEBaB/CEBaB.sa.${num_of_class}-class.${condition} \
#                     --do_train \
#                     --per_device_train_batch_size 32 \
#                     --max_source_length 128 \
#                     --learning_rate 2e-5 \
#                     --output_dir ./t5_cebab_${condition}_results/ \
#                     --save_total_limit 1 \
#                     --cache_dir ../huggingface_cache/ \
#                     --seed ${seed} \
#                     --report_to none \
#                     --logging_steps 10
#                 done
#             done
#         done
#     done
# done

# Challenge Set Experiments
# for condition in exclusive inclusive; do
#     for task_name in opentable; do
#         for seed in 42 66 77 88 99; do
#             for num_of_class in 2; do
#                 for model_name in bert-base-uncased; do
#                     # training script for the no-confounding case
#                     CUDA_VISIBLE_DEVICES=5,6,8,9 python run_cebab.py \
#                     --model_name_or_path ${model_name} \
#                     --task_name ${task_name} \
#                     --dataset_name CEBaB/CEBaB-challenge.sa.${num_of_class}-class.${condition} \
#                     --do_train \
#                     --do_eval \
#                     --max_seq_length 128 \
#                     --per_device_train_batch_size 32 \
#                     --per_device_eval_batch_size 32 \
#                     --learning_rate 2e-5 \
#                     --output_dir ./cebab_challenge_${condition}_results/ \
#                     --save_total_limit 1 \
#                     --cache_dir ../huggingface_cache/ \
#                     --seed ${seed} \
#                     --report_to none \
#                     --logging_steps 10
#                 done
#             done
#         done
#     done
# done

# SA Experiments
# for task_name in opentable; do
#     for seed in 42 66 77 88 99; do
#         for num_of_class in 2 3 5; do
#             for model_name in bert-base-uncased roberta-base gpt2; do
#                 # training script for the no-confounding case
#                 CUDA_VISIBLE_DEVICES=0,1,3,5 python run_cebab.py \
#                 --model_name_or_path ${model_name} \
#                 --task_name ${task_name} \
#                 --dataset_name CEBaB/CEBaB.sa.${num_of_class}-class.exclusive \
#                 --do_train \
#                 --max_seq_length 128 \
#                 --per_device_train_batch_size 32 \
#                 --per_device_eval_batch_size 32 \
#                 --learning_rate 2e-5 \
#                 --output_dir ./cebab_exclusive_results/ \
#                 --save_total_limit 1 \
#                 --cache_dir ../huggingface_cache/ \
#                 --seed ${seed} \
#                 --report_to none \
#                 --logging_steps 10
#             done
#         done
#     done
# done

# for task_name in opentable; do
#     for seed in 42 66 77 88 99; do
#         for num_of_class in 2 3 5; do
#             for model_name in lstm; do
#                 # training script for the no-confounding case
#                 CUDA_VISIBLE_DEVICES=0,1,3,5 python run_cebab.py \
#                 --model_name_or_path ${model_name} \
#                 --task_name ${task_name} \
#                 --dataset_name CEBaB/CEBaB.sa.${num_of_class}-class.exclusive \
#                 --do_train \
#                 --max_seq_length 128 \
#                 --per_device_train_batch_size 32 \
#                 --per_device_eval_batch_size 32 \
#                 --learning_rate 0.001 \
#                 --output_dir ./cebab_exclusive_results/ \
#                 --save_total_limit 1 \
#                 --cache_dir ../huggingface_cache/ \
#                 --seed ${seed} \
#                 --report_to none \
#                 --logging_steps 10
#             done
#         done
#     done
# done

# ABSA Experiments
# for task_name in opentable-absa; do
#     for seed in 42 66 77 88 99; do
#         for model_name in bert-base-uncased roberta-base gpt2; do # lstm; do
#             # training script for the no-confounding case
#             CUDA_VISIBLE_DEVICES=0,1,3,5 python run_cebab.py \
#             --model_name_or_path ${model_name} \
#             --task_name ${task_name} \
#             --dataset_name CEBaB/CEBaB.absa.inclusive \
#             --do_train \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --learning_rate 2e-5 \
#             --output_dir ./cebab_absa_inclusive_results/ \
#             --save_total_limit 1 \
#             --cache_dir ../huggingface_cache/ \
#             --seed ${seed} \
#             --report_to none \
#             --logging_steps 10
#         done
#     done
# done

# for task_name in opentable-absa; do
#     for seed in 42 66 77 88 99; do
#         for model_name in lstm; do
#             # training script for the no-confounding case
#             CUDA_VISIBLE_DEVICES=0,1,3,5 python run_cebab.py \
#             --model_name_or_path ${model_name} \
#             --task_name ${task_name} \
#             --dataset_name CEBaB/CEBaB.absa.inclusive \
#             --do_train \
#             --max_seq_length 128 \
#             --per_device_train_batch_size 32 \
#             --learning_rate 0.001 \
#             --output_dir ./cebab_absa_inclusive_results/ \
#             --save_total_limit 1 \
#             --cache_dir ../huggingface_cache/ \
#             --seed ${seed} \
#             --report_to none \
#             --logging_steps 10
#         done
#     done
# done