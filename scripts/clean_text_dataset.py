from vllm import LLM, SamplingParams
from datasets import Dataset,DatasetDict,concatenate_datasets,load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from termcolor import colored

import tempfile
import pandas as pd
import torch
import os
import shutil
import logging
import math

import config

def color_text(text,color='green',background='on_white'):
    return colored(text,color,background)

def generate_text(batch,llm,sampling_params,system_role,tokenizer):
    messages = [[
            {"role": "system", "content": system_role},
            {"role": "user", "content": f'Clean the following: {article}'}
        ] for article in batch["metadata"]]
    
    messages=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)
    
    outputs = llm.generate(messages, sampling_params)
    
    print(color_text(f"Batch size: {len(batch['metadata'])}"))
    print(color_text(f'Output size: {len(outputs)}'))
    assert len(batch['metadata'])==len(outputs), "LLM generates less response!"

    batch["clean_data"] = [output.outputs[0].text for output in outputs]
    return batch

def store_dataset(dataset,store_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset.save_to_disk(tmp_dir)
        shutil.rmtree(store_path,ignore_errors=True)
        shutil.move(tmp_dir, store_path)

def clean_text(text_list,system_role,model_id,seed=42,n_gpu=4,batch_size=128,path='../data/impresso/fullset'):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    dataset=Dataset.from_dict({'metadata':text_list})
    
    os.makedirs(path,exist_ok=True)
    processed_path=f'{path}/processed_batch_{get_subfix(system_role)}'
    is_processed=os.path.exists(processed_path)
    if is_processed:
        print(f"path {processed_path}")
        processed_dataset=load_from_disk(processed_path)
        processed_len=len(processed_dataset)
        whole_len=len(dataset)
        if processed_len<whole_len:
            dataset=dataset.select(range(processed_len,whole_len))
            print(color_text(f'{processed_len} samples have been processed, and {whole_len-processed_len} samples remaining.'))
        else:
            print(color_text('Whole dataset has been processed. Terminated.'))
            return processed_dataset            

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = LLM(model=model_id,tensor_parallel_size=n_gpu,max_model_len=45000)
    sampling_params = SamplingParams(temperature=0.5, top_p=.7,repetition_penalty=1.05, max_tokens=45000,seed=seed)
    
    processed_batches=[]
    for i, batch in tqdm(enumerate(dataset.iter(batch_size=batch_size))):
        processed_batch = Dataset.from_dict(batch).map(generate_text, batched=True,batch_size=batch_size,num_proc=1,fn_kwargs={'llm':llm,'sampling_params':sampling_params,'system_role':system_role,'tokenizer':tokenizer})
        processed_batches.append(processed_batch)
        print(color_text(f'Samples processed: {sum(map(len,processed_batches))}/{len(dataset)}'))
        
        if (i+1)%5==0 or (i+1)==math.ceil(len(dataset)/batch_size):
            if is_processed:
                generated_dataset=concatenate_datasets([processed_dataset]+processed_batches)
                print(color_text(f'CONTINUED from {processed_len} and {sum(map(len,processed_batches))} more samples saved.'))
            else:
                generated_dataset=concatenate_datasets(processed_batches)
                print(color_text(f"Total samples saved: {len(generated_dataset)} for INITIAL processing"))
            store_dataset(generated_dataset,processed_path)

    #generated_dataset = dataset.map(generate_text, batched=True, batch_size=batch_size,fn_kwargs={'llm':llm,'sampling_params':sampling_params,'system_role':system_role,'tokenizer':tokenizer})

    return generated_dataset

def get_subfix(role):
    if role==config.system_role_editor:
        subfix='clean'
    elif role==config.system_role_summarizer:
        subfix='sum'
    else:
        subfix='translate'
    return subfix

def remove_sprcial_chars(df_col):
    return df_col.fillna('Null').str.replace(r'[^a-zA-Z0-9\u00C0-\u017F\s.,]','',regex=True).replace(r"\s+",' ',regex=True).replace(r'\.{2,}', '.', regex=True).replace(r',{2,}', ',', regex=True).fillna('Null')

def main(system_role,model_id,path='../data/impresso',batch_size=256):
    n_gpu=torch.cuda.device_count()
    print(color_text(f'Current number of GPUs: {n_gpu}'))
    #out_path=f'{path}/fullset/impresso160k_{get_subfix(system_role)}'
    df=pd.read_csv(f'{path}/impresso160k.csv',sep=';',header=0,encoding='utf-8')
    generated_dataset=clean_text(remove_sprcial_chars(df['content']).tolist(),system_role,model_id,n_gpu=n_gpu,batch_size=batch_size*n_gpu,path=path+'/fullset')

if __name__=='__main__':
    role=config.system_role_editor
    model=config.qwen
    logging.basicConfig(level=logging.DEBUG)
    main(role,model)
