## BERT
# binary
# python main.py --output_dir output --flush_cache true --model_architecture bert-base-uncased --task_name opentable_binary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture bert-base-uncased --task_name opentable_binary --eval_split dev

# ternary
python main.py --output_dir output --flush_cache true --model_architecture bert-base-uncased --task_name opentable_ternary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture bert-base-uncased --task_name opentable_ternary --eval_split dev

# 5-way
python main.py --output_dir output --flush_cache true --model_architecture bert-base-uncased --task_name opentable_5_way --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture bert-base-uncased --task_name opentable_5_way --eval_split dev

## RoBERTa
# binary
python main.py --output_dir output --flush_cache true --model_architecture roberta-base --task_name opentable_binary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture roberta-base --task_name opentable_binary --eval_split dev

# ternary
python main.py --output_dir output --flush_cache true --model_architecture roberta-base --task_name opentable_ternary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture roberta-base --task_name opentable_ternary --eval_split dev

# 5-way
python main.py --output_dir output --flush_cache true --model_architecture roberta-base --task_name opentable_5_way --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture roberta-base --task_name opentable_5_way --eval_split dev

## GPT2
# binary
python main.py --output_dir output --flush_cache true --model_architecture gpt2 --task_name opentable_binary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture gpt2 --task_name opentable_binary --eval_split dev

# ternary
python main.py --output_dir output --flush_cache true --model_architecture gpt2 --task_name opentable_ternary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture gpt2 --task_name opentable_ternary --eval_split dev

# 5-way
python main.py --output_dir output --flush_cache true --model_architecture gpt2 --task_name opentable_5_way --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture gpt2 --task_name opentable_5_way --eval_split dev

## T5
## binary
# python main.py --output_dir output --flush_cache true --model_architecture t5-base --task_name opentable_binary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture t5-base --task_name opentable_binary --eval_split dev

## ternary
# python main.py --output_dir output --flush_cache true --model_architecture t5-base --task_name opentable_ternary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture t5-base --task_name opentable_ternary --eval_split dev

## 5-way
# python main.py --output_dir output --flush_cache true --model_architecture t5-base --task_name opentable_5_way --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture t5-base --task_name opentable_5_way --eval_split dev

## LSTM
# binary
python main.py --output_dir output --flush_cache true --model_architecture lstm --task_name opentable_binary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture lstm --task_name opentable_binary --eval_split dev

# ternary
python main.py --output_dir output --flush_cache true --model_architecture lstm --task_name opentable_ternary --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture lstm --task_name opentable_ternary --eval_split dev

# 5-way
python main.py --output_dir output --flush_cache true --model_architecture lstm --task_name opentable_5_way --eval_split test
# python main.py --output_dir output --flush_cache true --model_architecture lstm --task_name opentable_5_way --eval_split dev
