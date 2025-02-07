from vllm import LLM, SamplingParams
from datasets import Dataset
import pandas as pd
import torch

import config

def generate_text(batch,llm,sampling_params,system_role):
    messages = [[
            {"role": "system", "content": system_role},
            {"role": "user", "content": f'Clean the following: \n\n{article}'}
        ] for article in batch["metadata"]]

    outputs = llm.generate(messages, sampling_params)
    
    batch["clean_data"] = [output.outputs[0].text for output in outputs]
    return batch

def clean_impresso160k(text_list,system_role,model_id,seed=42,n_gpu=4,batch_size=128):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset=Dataset.from_dict({'metadata':text_list})

    llm = LLM(model=model_id,tensor_parallel_size=n_gpu,max_model_len=3076)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2000,seed=seed)

    generated_dataset = dataset.map(generate_text, batched=True, batch_size=batch_size,fn_kwargs={'llm':llm,'sampling_params':sampling_params,'system_role':system_role})
    
    return generated_dataset

def main(system_role,model_id,path='../data/impresso'):
    n_gpu=torch.cuda.device_count()
    df=pd.read_csv(f'{path}/impresso160k.csv',sep=';',header=0,encoding='utf-8')
    generated_dataset=clean_impresso160k(df['content'].fillna('Null').tolist(),system_role,model_id,n_gpu=n_gpu,batch_size=64*n_gpu)
    generated_dataset.save_to_disk(f'{path}/impresso160k_clean')

if __name__=='__main__':
    main(config.system_role_editor,config.qwen)
