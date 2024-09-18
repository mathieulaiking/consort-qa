methods=(0-shot few-shot fs-cot 1-shot-cot-orig)
train_path=../data/train.jsonl
test_path=../data/test.jsonl
n_shot=5

for method in "${methods[@]}"
do
    echo "Creating few-shot prompts with $method prompting"
    python create_prompts.py $train_path $test_path \
        --method $method \
        --n_shot $n_shot \
        --output_dir ./$method \
        --overwrite_output_dir
done