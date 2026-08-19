import argparse
import torch
import math
import json
import os
import re
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from datasets import load_from_disk
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from read_posters import get_all_poster
from poster_manipulation import extract_features
from feature_extractor import extract_feat_blip, extract_feat_mclip

import config

def get_data(type,path='../data/impresso/fullset'):
    if type.lower()=='clean':
        article=load_from_disk(f'{path}/processed_batch_clean')
    elif type.lower()=='sum':
        article=load_from_disk(f'{path}/processed_batch_sum')
    elif type.lower()=='translate':
        article=load_from_disk(f'{path}/processed_batch_translate')
    else:
        raise Exception('No such a data type.')
    images,captions=get_all_poster()
    return images,article

def plot_txt2img(texts,retrieved_images,index):
    n_block=len(texts)
    n_col=len(retrieved_images)
    fig = plt.figure(figsize=(3*n_col, 5*n_block))
    main_gs = GridSpec(n_block, 1, figure=fig, hspace=0.05)  # Controls spacing BETWEEN blocks

    for i in range(n_block):
        block_subgs = main_gs[i].subgridspec(2, n_col, hspace=0, wspace=0,height_ratios=[2,3])

        ax_top = fig.add_subplot(block_subgs[0, :])
        #ax_top.set_facecolor('#EEEEEE')

        ax_top.text(.01,.5,textwrap.fill(texts[i], width=120),ha='left',va='center',transform=ax_top.transAxes,fontsize=15,fontstyle='italic')
        
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ['top','bottom','left','right']:
            ax_top.spines[spine].set_visible(False)

        ax_top.set_ylabel(f'Query {i+1}',fontdict=dict(weight='bold',size=18))

        for j in range(n_col):
            ax = fig.add_subplot(block_subgs[1,j])
            ax.imshow(retrieved_images[i][j])
            #ax.set_title(f"Plot {col+1}")
            ax.set_xticks([])
            ax.set_yticks([])
            if index[i][j] == 1:
                for spine in ax.spines.values():
                    spine.set_color('green')
                    spine.set_linewidth(5)

    margin_config={
                    'left':0.05,
                    'right':.95,
                    'bottom':0.01,
                    'top':.99
                    }
    plt.subplots_adjust(**margin_config) 
    
    return fig

def plot_img2txt(images,retrieved_texts,index):
    n_block=len(images)
    n_col=6
    fig = plt.figure(figsize=(3*(n_col+1), 6*n_block))
    main_gs = GridSpec(n_block, 1, figure=fig, hspace=0.03)  # Controls spacing BETWEEN blocks

    for i in range(n_block):
        width_ratios=[2]+[1]*(n_col-1)
        block_subgs = main_gs[i].subgridspec(5, n_col, hspace=0, wspace=0, width_ratios=width_ratios)
        
        for j in range(n_col):
            if j==0:
                ax= fig.add_subplot(block_subgs[:,j])
                ax.imshow(images[i])
                
                for spine in ['top','bottom','left','right']:
                    ax.spines[spine].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(f'Query {i+1}',fontdict=dict(weight='bold',size=18))
            else:
                ax= fig.add_subplot(block_subgs[j-1,1:])
                ax.text(.01,.5,textwrap.fill(retrieved_texts[i][j-1], width=120),ha='left',va='center',transform=ax.transAxes,fontsize=15,fontstyle='italic',color='green' if index[i][j-1]==1 else 'black')
                ax.set_xticks([])
                ax.set_yticks([])
               
                ax.spines['left'].set_edgecolor('gray')
                for spine in ['top', 'bottom', 'right','left']:
                    ax.spines[spine].set_visible(False)
                
                if j!=n_col-1:
                    ax.axhline(y=0, color='gray', linewidth=1, xmin=0, xmax=1)

    margin_config={
                    'left':0.05,
                    'right':.95,
                    'bottom':0.01,
                    'top':.99
                    }
    plt.subplots_adjust(**margin_config)

    return fig

def cut_caps(string,threshold=5):
    tokens=string.strip().split(' ')
    if tokens[-1]=='.':
        tokens[-2]+='.'
        tokens.pop()
    leng=len(tokens)
    res=''
    cnt=0
    if leng>50:
        tokens=tokens[:50]

    for token in tokens:
        cnt+=1
        if cnt==leng:
            res+=token
        else:
            res+=token+' '
        if cnt%threshold==0 and cnt!=leng:
            res+='\n'
    if leng>50:
        res+=' [...]'
    return res

def cut_caps2(string):
    tokens=string.strip().split(' ')
    if tokens[-1]=='.':
        tokens[-2]+='.'
        tokens.pop()
    leng=len(tokens)
    res=''
    cnt=0
    if leng>50:
        tokens=tokens[:50]

    for token in tokens:
        cnt+=1
        if cnt==leng:
            res+=token
        else:
            res+=token+' '
    if leng>50:
        res+=' [...]'
    return res

def get_expert_labels(model,task):
    if task=='img2txt':
        if model.lower()=='clip':
            index = config.i2t_clip 
        elif model.lower()=='blip':
            index = config.i2t_blip
        elif model.lower()=='mclip':
            index = config.i2t_mclip
        else:
            raise Exception('No such a model.')
    elif task=='txt2img':
        if model.lower()=='clip':
            index = config.t2i_clip 
        elif model.lower()=='blip':
            index = config.t2i_blip
        elif model.lower()=='mclip':
            index = config.t2i_mclip
        else:
            raise Exception('No such a model.')
    else:
        raise Exception('No such a task.')
    return index

def plot_examples(model,task,images,texts,k=5,seed=42,device=0):
    features=extract_features({'images':images,'texts':texts},model,device=device)
    img_embed,cap_embed=features['image features'].cpu(),features['text features'].cpu()
    if task=='txt2img':
        row_length=14
    else:
        row_length=13
    texts=list(map(cut_caps2,texts))

    cosine=pairwise_cosine_similarity(img_embed,cap_embed)
    if task!='img2txt':
        cosine=cosine.T

    vals,ids=torch.topk(cosine,k=k,dim=1)
    print(f"Average similarity for task {task}: {vals.mean(dim=1)}")
    print(f"Total average similarity: {vals.mean()}")
    
    if task=='img2txt':
        nrow=len(images)
        index = get_expert_labels(model, task)
        retrieved_items=[[texts[item] for item in items] for items in ids]
        fig=plot_img2txt(images,retrieved_items,index)
    else:
        nrow=len(texts)
        index = get_expert_labels(model, task)
        retrieved_items=[[images[item] for item in items] for items in ids]
        fig=plot_txt2img(texts,retrieved_items,index) 
    fig.savefig(f'../figures/retrieval_example_{model}_{task}_{seed}.pdf')
    print('DONE.')


def pick_top1(cosine):
    res=[]
    vals,ids=torch.topk(cosine,k=1,dim=1)
    for i,d in enumerate(ids):
        if d.item()<=3:
            res.append(i)
    return res

def get_index(model,images,texts,topics):
    text_ids=[]
    topics+=['other topics']
    
    cross_modal_featues=extract_features({'images':images,'texts':[f'a picture of {topic}' for topic in topics]},model)
    img_feats,txt_feats=cross_modal_featues['image features'],cross_modal_featues['text features']
    cosine=pairwise_cosine_similarity(img_feats,txt_feats)
    img_ids=pick_top1(cosine)
    
    size=1000
    for i in tqdm(range(math.ceil(len(texts)/size))):
        intra_modal_features=extract_features({'images':[images[0]],'texts':[f'an article of {topic}' for topic in topics]+texts[size*i:size*(i+1)]},model)
        inter_cosine=pairwise_cosine_similarity(intra_modal_features['text features'][len(topics):],intra_modal_features['text features'][:len(topics)])
        temp=[size*i+idx for idx in pick_top1(inter_cosine)]
        text_ids.extend(temp)

    return img_ids,text_ids
    

def params():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=str,help='clip or blip')
    parser.add_argument('--type',type=str,default='sum',help='clean, sum, translate')
    parser.add_argument('--device',type=int,default=0,help='device id')

    args=parser.parse_args()

    return args


if __name__=='__main__':
    n_examples = 5
    args=params()

    images,article=get_data(args.type)
    images=list(map(lambda img:(img*255).to(torch.uint8).permute(1,2,0),images))
    
    if args.model.lower()=='clip':
        seed=6
    elif args.model.lower()=='blip':
        seed=8
    elif args.model.lower()=='mclip':
        seed=42

    path=f'../data/index_{args.model.lower()}.json'
    if os.path.exists(path):
        with open(path,'r') as f:
            indices=json.load(f)
            img_ids,txt_ids=indices['img_ids'],indices['txt_ids']
    else:
        img_ids,txt_ids=get_index(args.model,images,article['clean_data'],config.english_topics)
        with open(path,'w') as f:
            json.dump({'img_ids':img_ids,'txt_ids':txt_ids},f)

    discard_img_ids=[idx for idx in range(len(images)) if idx not in img_ids]
    discard_txt_ids=[idx for idx in range(len(article['clean_data'])) if idx not in txt_ids]

    print(f"Retained {len(img_ids)} images and {len(txt_ids)} texts.")
    print(f"Discarded {len(discard_img_ids)} images and {len(discard_txt_ids)} texts.")

    
    for task in ['img2txt','txt2img']:
        if task=='img2txt':
            np.random.seed(6*seed)
            rand_img_ids=np.random.choice(img_ids,n_examples,replace=False)
            txt_ids = discard_txt_ids #for discarded samples
            rand_txt_ids=np.random.choice(txt_ids,1000,replace=False)
            article_subset = [re.sub(r'[\r\n]+',' ',article['clean_data'][int(idx)]) for idx in rand_txt_ids]
        else:
            np.random.seed(8*seed)
            rand_txt_ids=np.random.choice(txt_ids,n_examples,replace=False)
            img_ids = discard_img_ids # for discarded samples
            try:
                rand_img_ids=np.random.choice(img_ids,1000,replace=False)
            except ValueError:
                rand_img_ids=np.random.choice(img_ids,len(img_ids),replace=False)
            subset = [re.sub(r'[\r\n]+',' ',article['metadata'][int(idx)]) for idx in rand_txt_ids]
            tokenizer = AutoTokenizer.from_pretrained(config.qwen)
            qwen_model = AutoModelForCausalLM.from_pretrained(config.qwen,
                                                            device_map=f"cuda:{args.device}",
                                                            torch_dtype=torch.float16)
            messages = [[
                {"role": "system", "content": config.system_role_translator},
                {"role": "user", "content": f'Translate the following to English: {article}'}] for article in subset]
            article_subset = []
            
            for message in tqdm(messages):
                text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer(text, return_tensors="pt", padding=True).to(qwen_model.device)
                generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=45000)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                article_subset.append(response)
                # print(response)
                
        plot_examples(args.model,task,[images[int(idx)] for idx in rand_img_ids],article_subset,seed=seed,device=args.device)