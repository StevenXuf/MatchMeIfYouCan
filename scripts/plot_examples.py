import argparse
import torch
import math
import json
import os
import re
import textwrap
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from datasets import Dataset,load_from_disk
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from read_posters import get_all_poster
from poster_manipulation import extract_features

import config

def get_data(path='../data/impresso/fullset'):
    images,captions=get_all_poster()
    article_clean=load_from_disk(f'{path}/processed_batch_clean')['clean_data']
    article_sum=load_from_disk(f'{path}/processed_batch_sum')['clean_data']
    return images,article_clean,article_sum 

def plot_txt2img(texts,retrieved_images):
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
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.axis('off')
    margin_config={
                    'left':0.05,
                    'right':.95,
                    'bottom':0.01,
                    'top':.99
                    }
    plt.subplots_adjust(**margin_config) 
    
    return fig

def plot_img2txt(images,retrieved_texts):
    n_block=len(images)
    n_col=6
    fig = plt.figure(figsize=(3*n_col, 6*n_block))
    main_gs = GridSpec(n_block, 1, figure=fig, hspace=0.03)  # Controls spacing BETWEEN blocks

    for i in range(n_block):
        block_subgs = main_gs[i].subgridspec(5, n_col, hspace=0, wspace=0)
        
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
                ax.text(.01,.5,textwrap.fill(retrieved_texts[i][j-1], width=120),ha='left',va='center',transform=ax.transAxes,fontsize=15,fontstyle='italic')
                ax.set_xticks([])
                ax.set_yticks([])
               
                ax.spines['left'].set_edgecolor('gray')
                for spine in ['top', 'bottom', 'right']:
                    ax.spines[spine].set_visible(False)
                
                if j!=n_col-1:
                    ax.axhline(y=0, color='gray', linewidth=1,xmin=0, xmax=1)

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

def plot_examples(model,task,images,texts,k=5):
    features=extract_features({'images':images,'texts':texts},model)
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
    
    if task=='img2txt':
        nrow=len(images)
        retrieved_items=[[texts[item] for item in items] for items in ids]
        fig=plot_img2txt(images,retrieved_items)
    else:
        nrow=len(texts)
        retrieved_items=[[images[item] for item in items] for items in ids]
        fig=plot_txt2img(texts,retrieved_items) 
    fig.savefig(f'../figures/retrieval_example_{model}_{task}.pdf')
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
    
    size=10000
    for i in tqdm(range(math.ceil(len(texts)/size))):
        intra_modal_features=extract_features({'images':[images[0]],'texts':[f'an article of {topic}' for topic in topics]+texts[size*i:size*(i+1)]},model)
        inter_cosine=pairwise_cosine_similarity(intra_modal_features['text features'][len(topics):],intra_modal_features['text features'][:len(topics)])
        temp=[size*i+idx for idx in pick_top1(inter_cosine)]
        text_ids.extend(temp)

    return img_ids,text_ids
    

def params():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=str,help='clip or blip')
    args=parser.parse_args()

    return args

if __name__=='__main__':
    n_examples=5
    torch.manual_seed(42)

    args=params()
    images,article_clean,article_sum=get_data()
    images=list(map(lambda img:(img*255).to(torch.uint8).permute(1,2,0),images))
    
    if args.model.lower()=='clip':
        seed=6
    else:
        seed=8
   
    path=f'../data/index_{args.model.lower()}.json'
    if os.path.exists(path):
        with open(path,'r') as f:
            indices=json.load(f)
            img_ids,txt_ids=indices['img_ids'],indices['txt_ids']
    else:
        img_ids,txt_ids=get_index(args.model,images,article_sum,config.english_topics)
        with open(path,'w') as f:
            json.dump({'img_ids':img_ids,'txt_ids':txt_ids},f)

    print(len(img_ids))
    print(len(txt_ids))
    
    for task in ['img2txt','txt2img']:
        if task=='img2txt':
            np.random.seed(6*seed)
            rand_img_ids=np.random.choice(img_ids,n_examples,replace=False)
            rand_txt_ids=np.random.choice(txt_ids,1000,replace=False)
        else:
            np.random.seed(8*seed)
            rand_img_ids=np.random.choice(img_ids,1000,replace=False)
            rand_txt_ids=np.random.choice(txt_ids,n_examples,replace=False)

        plot_examples(args.model,task,[images[idx] for idx in rand_img_ids],[re.sub(r'[\r\n]+',' ',article_sum[idx]) for idx in rand_txt_ids])
