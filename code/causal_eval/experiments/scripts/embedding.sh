for task_name in opentable_binary; do  # opentable_ternary opentable_5_way; do
    for model_name in bert-base-uncased; do  # roberta-base t5-base gpt2 lstm; do
        python methods/embedding.py \
            --model_name_or_path ${model_name} \
            --task_name ${task_name} \
            --embeddings_output_dir ./outputs_embeddings/untrained/${task_name}/${model_name}/None__None
    done
done