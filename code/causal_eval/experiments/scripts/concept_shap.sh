for seed in 42 43 44 45 46; do
    for task_name in opentable_binary; do  # opentable_ternary opentable_5_way; do
        for model_name in bert-base-uncased; do  # roberta-base t5-base gpt2 lstm; do
            instance_path="${task_name}/${model_name}/None__None/seed_${seed}"
            python concept_shap.py \
                --seed ${seed} \
                --task_name ${task_name} \
                --concepts ambiance food noise service \
                --model_path ./outputs_run_opentable/${instance_path} \
                \
                --cavs_output_dir ./outputs_cavs/${instance_path} \
                --embeddings_output_dir ./outputs_embeddings/trained/${instance_path} \
                --concept_shap_output_dir ./outputs_concept_shap_scores/${instance_path} \
                --verbose True
        done
    done
done