import os
import torch
import argparse
import fire
import pandas as pd

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from datasets import load_from_disk

import config
from poster_manipulation import get_poster_subset,extract_features,get_precision_recall
from text_manipulation import get_contents,denest
from read_posters import get_all_poster
from clean_text_dataset import clean_text

def compute_poster_caption_metrics(features,task='img2txt'):
    sim=pairwise_cosine_similarity(features['image features'].cpu(),features['text features'].cpu())
    targets=torch.eye(sim.size(0)).long()

    if task=='txt2img':
        sim=sim.T
    print('*'*10+task+'*'*10)
    
    assert sim.size()==targets.size(),'Should be the same size.'

    for k in [1,5,10]:
        get_precision_recall(sim,targets,k)

def params():
    parser = argparse.ArgumentParser(description="compute recall and precision")
    parser.add_argument('model')
    parser.add_argument('-t','--task',type=str,default='img2txt',help='img2txt or txt2img?')
    args = parser.parse_args()
    return args

def get_poster_metrics(is_translated=False):
    args=params()
    images,captions=get_all_poster()
    images=list(map(lambda img:(img*255).to(torch.uint8).permute(1,2,0),images))
    captions=list(map(lambda x: x.strip(' ').split(';')[0],captions))
    print(len(captions), len(images))

    if is_translated:
        path='../data/LAKA_caps'
        if not os.path.exists(path):
            dataset=clean_text(captions,config.system_role_translator,config.qwen,seed=42,n_gpu=1,batch_size=256,path=path)
        else:
            print(f"Loading translated captions from {path}")
            dataset_path=f'{path}/processed_batch_translate'
            dataset = load_from_disk(dataset_path)
            #map(lambda x: {'metadata': x['metadata'],
            #                                               #   'clean_data': x['clean_data']})
        df = dataset.to_pandas()
        print(df.shape)
        df.set_index('metadata',inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        captions = df.loc[captions]['clean_data'].tolist()

    print(len(captions)==len(images))
    features=extract_features({'images':images,'texts':captions},args.model)
    compute_poster_caption_metrics(features,args.task)

def get_targets(poster,articles,task):
    df=poster['anno']
    poster_labels=torch.tensor(df.values)
    article_cnts=list(map(len,articles))
    article_labels=torch.zeros(sum(article_cnts),poster_labels.size(1))
    
    start_idx=0
    for i in range(poster_labels.size(1)):
        article_labels[start_idx:start_idx+article_cnts[i],i]=torch.ones(article_cnts[i],)
        start_idx+=article_cnts[i]
    article_labels=article_labels.to(torch.int64)
    
    if task=='img2txt':
        targets=torch.matmul(poster_labels,torch.transpose(article_labels,0,1))
    elif task=='txt2img':
        targets=torch.matmul(article_labels,torch.transpose(poster_labels,0,1))
    
    targets=targets>=1
    print(targets.size())
    
    return targets.long()


def compute_poster_article_metrics(system_role,model_name,task,llm=config.qwen,is_meta=False):
    _,articles,titles=get_contents(config.english_topics,system_role,llm,is_meta=is_meta)
    
    poster=get_poster_subset(config.in_file,config.out_file,config.anno_file)
    features=extract_features({'images':poster['images'],'texts':denest(articles)},model_name)
    features['image features']=features['image features'].cpu()
    features['text features']=features['text features'].cpu()
    
    print('*'*10+task+'*'*10)
    if task=='img2txt':
        preds=pairwise_cosine_similarity(features['image features'],features['text features'])
    elif task=='txt2img':
        preds=pairwise_cosine_similarity(features['text features'],features['image features'])
    targets=get_targets(poster,articles,task)
    
    for k in [1,5,10]:
        get_precision_recall(preds,targets,k)
    
def main(system_role,model_name,task,meta=False):
    if system_role == 'translator':
        system_role=config.system_role_translator
    elif system_role == 'editor':
        system_role=config.system_role_editor
    elif system_role == 'summarizer':
        system_role=config.system_role_summarizer

    print(system_role)
    compute_poster_article_metrics(system_role,model_name,task,is_meta=meta)

if __name__=='__main__':
    #experiments for laka images and impresso articles
    # fire.Fire(main)


    #experiments for laka images and topics/captions
    get_poster_metrics(is_translated=True)
