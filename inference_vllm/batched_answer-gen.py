import os
import json
import argparse
import logging
import pandas as pd
import sklearn.metrics as sklm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
# try/except custom library import removed for anonymity
def log_info_rank0(*msg):
    logging.info(*msg)

def parse_args():
    argument_parser = argparse.ArgumentParser(
        description="""This program performs chain-of-thought on a collection of prompts for CONSORT-QA, these prompts should be prompts for CoT:
        for each prompt it will perform an explanation generation and then generate the Yes/No answer ;
        it is batched to avoid ray TimeOut Error that we encountered with vLLM v0.2.3 when sending too much prompts at once ;
        it's possible to resume the generations as each batch is checkpointed (text and predictions are saved) ;
        name of each prompt file in `prompt_dir` should be : <article_id>_<consort_id>.txt"""
    )
    argument_parser.add_argument("prompt_dir",type=str,help="path to directory containing prompts (dir with .txt files inside)")
    argument_parser.add_argument("model_path",type=str,help="path to base pretrained model")
    argument_parser.add_argument("output_dir", type=str,help="path to output directory")
    argument_parser.add_argument("--batch_size", type=int, default=64, help="size of prompt batches given to vllm generate function")
    argument_parser.add_argument("--n_gpus", type=int, default=None, help="number of gpus to use")
    argument_parser.add_argument("--dtype", type=str, default='auto', help="data type for model weights and activations : 'auto','float32' , 'float16' or 'bfloat16'. If `auto`, we use the `torch_dtype` attribute specified in the model config file")
    args = argument_parser.parse_args()
    log_info_rank0("args : "+ str(vars(args)))
    
    # verifications
    if not os.path.isdir(args.prompt_dir) :
        raise ValueError(args.prompt_dir + " does not exist or is not a directory")
    if args.output_dir is not None and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        log_info_rank0("output_dir already exists, will resume generation with missing prompt files")
    if not os.path.isdir(args.model_path) :
        raise ValueError(args.model_path + " does not exist or is not a directory")
    return args

def prompt_batch_generator(files, filedir, batch_size=64):
    filename_batch = []
    text_batch = []
    last_file_index = len(files)
    for i,f in enumerate(files) :
        if not f.endswith(".txt"):
            continue
        text = open(os.path.join(filedir,f)).read()
        text_batch.append(text)
        filename_batch.append(f)
        if len(filename_batch) == batch_size or i+1==last_file_index:
            yield text_batch,filename_batch
            filename_batch = []
            text_batch = []

def classification_report(pred,true, average="macro"):
    return {
        "accuracy" : sklm.accuracy_score(y_true=true, y_pred=pred),
        "f1-score" : sklm.f1_score(y_true=true, y_pred=pred, average=average),
        "precision" : sklm.precision_score(y_true=true, y_pred=pred, average=average, zero_division=0.0),
        "recall" : sklm.recall_score(y_true=true, y_pred=pred, average=average, zero_division=0.0)
    }

def main():
    args = parse_args()
    
    # Load data
    files = [f for f in os.listdir(args.prompt_dir) if f.endswith(".txt")]
    
    # init predictions dict and output txt dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.output_dir, "predictions.json")):
        pred_dict = {}
    else :
        pred_dict = json.load(open(os.path.join(args.output_dir, "predictions.json")))
    
    # Remove already generated files
    if len(pred_dict) < len(files):
        files = [f for f in files if f not in pred_dict]
        log_info_rank0(f"resuming generation with missing prompt files, remaining : {len(files)}")
        
    # Load model
    model = LLM(
        args.model_path,
        tensor_parallel_size=args.n_gpus,
        max_logprobs=20,
        dtype=args.dtype, # bfloat16 for A100, float16 for V100
    )

    # Get tokenizer with yes and no tokens ids
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    yes_tokens = tokenizer.encode("yes Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no No", add_special_tokens=False)

    # batched generation
    for text_batch, filename_batch in prompt_batch_generator(files, args.prompt_dir, args.batch_size) :
        # Answer generation (Yes/No)
        ans_request_outputs = model.generate(
            prompts = text_batch,
            sampling_params=SamplingParams(
                temperature=0,
                logprobs=20,
                max_tokens=1,
            ),
            use_tqdm=False,
        )

        # Prediction Parsing
        for filename, request_output in zip(filename_batch, ans_request_outputs):
            # parse_prediction in logprobs
            prediction=-1
            for lp in sorted(request_output.outputs[0].logprobs[0].values(), key=lambda x: x.rank):
                if lp.decoded_token.lower().strip() == "yes":
                    prediction="Yes"
                    break
                elif lp.decoded_token.lower().strip() == "no":
                    prediction="No"
                    break
            # assign prediction in df
            pred_dict[filename] = prediction

        # save predictions 
        with open(os.path.join(args.output_dir, "predictions.json"),"w") as pred_file:
            json.dump(pred_dict,pred_file)

if __name__ == "__main__":
    main()