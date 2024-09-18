# CONSORT-QA : Evaluating clinical trials abstract reporting quality with generative large language models

This repository contains the data and the code developped for our CONSORT-QA paper.

# Installation 
First, check the [prerequisites for installing vllm](https://docs.vllm.ai/en/latest/getting_started/installation.html).  
Then, to install the required packages to run our experiments, run `pip install -r requirements.txt`.

# Data
The [data](./data) directory contains our datasets (covid and depression). The depression corpus is split in two : depression-rct contains the 99 abstracts described in the paper, the depression-crt contains abstracts that have not been considered in the paper. 
* The [labels.csv](./data/covid/labels.csv) files contains the CONSORT labels.
* The [sentences.csv](./data/covid/sentences.csv) files contains the sentences extracted from the abstracts and their predicted relevant consort id (if there is one).
* The [questions_and_examples.csv](./data/covid/questions_and_examples.csv) files contains the consort questions reformulated and their associated example from the original CONSORT declarations
* For depression data, there is also an [original_labels.csv](./data/depression-rct/original_labels.csv) file that contains the original labels from the annotators.

# Prompting
the [`prompts`](./prompts) dir contains code to create the prompts dir for model inference from the CONSORT-QA data. To create them just run :
```bash
cd prompts
bash run_prompt_creation.sh
```

# Inference
The [`inference_vllm`](./inference_vllm) dir contains code to run the model inference using vllm. 

* For the simple answer generation (0-shot and few-shot prompts), you can use the [answer_gen](./inference_vllm/answer_gen.py) script, which will compute the whole dir at once, as recommended by vLLM docs for faster computation.  

* We used a [batched](./inference_vllm/batched_cot.py) version of the code for running chain-of-thought because of a Ray TimeOut error we encountered with vLLM 0.2.3 ; and so we had to run it and checkpoint the model generations after each batch, and re run on the remaining prompts if it timed out. If there is any crash during the generation, just use the same --output_dir argument and the code will resume from the last checkpoint.  

The arguments for generation are described in each python files and can be displayed with the `--help` argument

note : the answer generation sampling params in the inference files have been updated for further experiments but the results presented in the paper are obtained using the method described : greedy decoding and considering the 10 first tokens in the distribution (logprobs) when generating the answer token.