import os
import ast
import json
import pandas
import argparse
import jsonlines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_path",
        type=str,
        default=None,
        help="path to train.jsonl data"
    )
    parser.add_argument(
        "test_path",
        type=str,
        default=None,
        help="path to test.jsonl data"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cot",
        help="prompting method to use , choose between 0-shot, few-shot and cot"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=5,
        help="number of shots if method is few-shot or cot",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output path to directory where prompts will be saved, raise error if exists and not empty"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="will overwrite output directory if it exists"
    )
    args = parser.parse_args()

    # args checks
    if args.method not in ['0-shot', '1-shot-cot-orig', 'few-shot', 'fs-cot']:
        raise ValueError("Wrong method, choose between '0-shot', '1-shot-cot-orig', 'few-shot', 'fs-cot' ")
    if args.output_dir is not None and not args.overwrite_output_dir and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory already exists and is not empty, use --overwrite_output_dir to erase it")
    return args

_TASK_INSTRUCTIONS = "Instructions : The task is to verify a criterion from the Consolidated Standards of Reporting Trial (CONSORT) for a given abstract. The output should be yes or no (whether the criterion is met or not)."

_TRAIN_BASE_TEMPLATE = """Context : '''{context}'''.
Question : {question}
Answer : {answer}""" # only positive examples in experiments

_TRAIN_COT_TEMPLATE = """Context : '''{context}'''.
Question : {question}
Explanation : {explanation}
Answer : {answer}""" # only positive examples in experiments

_TEST_BASE_TEMPLATE = """Context : '''{context}'''.
Question : {question}
Answer : """

_TEST_COT_TEMPLATE = """Context : '''{context}'''.
Question : {question}
Explanation : """

_GLOBAL_TEMPLATE="{task_instructions}{train_examples}{test_example}"

def get_num(s:str):
    return int(''.join(c for c in s if c.isdigit()))

def main():
    args = parse_args()
    
    # prepare output directory
    output_dir = args.output_dir
    if args.method == "few-shot":
        output_dir = output_dir.replace(args.method,f"{args.n_shot}-shot")
    if args.method == "fs-cot":
        output_dir = output_dir.replace(args.method,f"{args.n_shot}-shot-cot")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) == 4272 :
        print(output_dir, " already exists and filled, skipping creation")
        return 
    else :
        os.mkdir(output_dir)
    
    with jsonlines.open(args.test_path) as test_set:
        for test_example in test_set:
            # load train examples
            if args.method != "1-shot-cot-orig" :
                train_examples = [t for t in jsonlines.open(args.train_path) if t["corpus"] == test_example["corpus"]]
                train_examples = train_examples[:min(args.n_shot,len(train_examples))]
            else :
                train_examples = [t for t in jsonlines.open(args.train_path) if t["corpus"] == "consort-abstract-examples"]
            for qid in test_example["questions"]:
                task_instructions = _TASK_INSTRUCTIONS + '\n\n'
                
                # fill in-context train examples
                train_text = "" # if 0-shot , we just leave train_text empty
                if args.method in ["few-shot","fs-cot"]:
                    for train_example in train_examples:
                        if args.method == "few-shot":
                            train_text += _TRAIN_BASE_TEMPLATE.format(
                                context=train_example['context'],
                                question=train_example['questions'][qid],
                                answer=train_example['answers'][qid]
                            ) + '\n\n'
                        elif args.method == "fs-cot":
                            train_text += _TRAIN_COT_TEMPLATE.format(
                                context=train_example['context'],
                                question=train_example['questions'][qid],
                                explanation=train_example['explanations'][qid],
                                answer=train_example['answers'][qid]
                            ) + '\n\n'
                
                elif args.method == "1-shot-cot-orig":
                    train_example = [e for e in train_examples if qid in e["questions"]][0]
                    train_text +=  _TRAIN_COT_TEMPLATE.format(
                        context=train_example['context'],
                        question=train_example['questions'][qid],
                        explanation=train_example['explanations'][qid],
                        answer=train_example['answers'][qid]
                    ) + '\n\n'
                
                # fill test example
                if args.method in ["0-shot","few-shot"]:
                    test_text = _TEST_BASE_TEMPLATE.format(
                        context=test_example["context"],
                        question=test_example["questions"][qid]
                    )
                elif args.method in ["1-shot-cot-orig", "fs-cot"]:
                    test_text = _TEST_COT_TEMPLATE.format(
                        context=test_example["context"],
                        question=test_example["questions"][qid]
                    )

                # fill global prompt
                prompt = _GLOBAL_TEMPLATE.format(
                    task_instructions=task_instructions,
                    train_examples=train_text,
                    test_example=test_text
                )
                
                # save prompt to output_dir
                file_id = test_example['id'].replace('/','@')
                output_prompt_path = os.path.join(output_dir,f"{file_id}_{qid}.txt")
                with open(output_prompt_path,'w') as f:
                    f.write(prompt)

if __name__ == '__main__' :
    main()