import torch
import argparse

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.classification import MulticlassPrecision,MulticlassRecall,MultilabelPrecision,MultilabelRecall

import config
from poster_manipulation import get_poster_subset,extract_features
from text_manipulation import get_contents,manipulate_texts

def compute_poster_metrics(features,task='img2txt'):
    sim=pairwise_cosine_similarity(features['image features'].cpu(),features['text features'].cpu())
    targets=torch.arange(features['image features'].size(0))

    if task=='txt2img':
        sim=torch.transpose(sim,0,1)
    print('*'*10+task+'*'*10)
    
    assert sim.size(0)==len(targets),'Should be the same size.'

    for k in [1,5,10]:
        precision_at_k=MulticlassPrecision(sim.size(1),top_k=k)
        recall_at_k=MulticlassRecall(sim.size(1),top_k=k)

        precision_val=precision_at_k(sim,targets)
        recall_val=recall_at_k(sim,targets)
        print(f'Top {k} precision: {precision_val.item()*100:.2f}')
        print(f'Top {k} recall: {recall_val.item()*100:.2f}')

def params():
    parser = argparse.ArgumentParser(description="compute recall and precision")
    parser.add_argument('model')
    parser.add_argument('-t','--task',type=str,default='img2txt',help='img2txt or txt2img?')
    args = parser.parse_args()
    return args

def get_poster_metrics():
    args=params()
    poster=get_poster_subset(config.in_file,config.out_file,config.anno_file)
    features=extract_features(poster,args.model)
    compute_poster_metrics(features,args.task)



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
    
    print(poster_labels)
    print(article_labels)

    if task=='img2txt':
        targets=torch.matmul(poster_labels,torch.transpose(article_labels,0,1))
    elif task=='txt2img':
        targets=torch.matmul(article_labels,torch.transpose(poster_labels,0,1))
    
    targets=targets>=1
    print(targets)
    
    for i in range(targets.size(0)):
        print(torch.sum(targets[i,:]))
    return targets.long()


def compute_poster_article_metrics(text_type,system_role,model_name,task,top_k=10):
    _,articles,titles=get_contents(config.english_topics)
    if text_type=='articles':
        corpus=articles
    elif text_type=='titles':
        corpus=titles
    else:
        raise Exception('No such a text type')
    '''
    corpus=manipulate_texts(
            config.llama_3_1_8b_instruct,
            system_role,
            corpus)
    '''
    corpus=[item for sublist in corpus for item in sublist]
    poster=get_poster_subset(config.in_file,config.out_file,config.anno_file)
    features=extract_features({'images':poster['images'],'texts':corpus},model_name)
    features['image features']=features['image features'].cpu()
    features['text features']=features['text features'].cpu()
    
    print('*'*10+task+'*'*10)
    if task=='img2txt':
        cosine=pairwise_cosine_similarity(features['image features'],features['text features'])
    elif task=='txt2img':
        cosine=pairwise_cosine_similarity(features['text features'],features['image features'])
    targets=get_targets(poster,articles,task)
   
    precision_at_k=MultilabelPrecision(num_labels=cosine.size(1))
    recall_at_k=MultilabelRecall(num_labels=cosine.size(1))
    
    preds_ids=torch.topk(cosine,top_k,dim=1).indices
    preds=torch.zeros_like(cosine,dtype=targets.dtype).scatter(1,preds_ids,1)

    precision_val=precision_at_k(preds,targets)
    recall_val=recall_at_k(preds,targets)

    print(f'Top {top_k} precision: {precision_val.item()*100:.2f}')
    print(f'Top {top_k} recall: {recall_val.item()*100:.2f}')
    
    pre=.0
    re=.0
    target_ids=torch.topk(targets,top_k,dim=1).indices
    for i in range(targets.size(0)):
        correct_ids=preds_ids[i][torch.isin(preds_ids[i],target_ids[i])]
        pre+=correct_ids.size(0)/top_k
        re+=correct_ids.size(0)/torch.sum(targets[i]) if torch.sum(targets[i])!=0 else 0
        print(correct_ids.size(0)/torch.sum(targets[i]))
    print(f'Avg Precision: {pre/targets.size(0)*100:.2f}')
    print(f'Avg Recall: {re/targets.size(0)*100:.2f}')

if __name__=='__main__':
    text_type='articles'
    system_role=config.system_role_en_summarizer
    task='img2txt'
    model_name='clip'
    top_k=config.top_k

    compute_poster_article_metrics(text_type,system_role,model_name,task,top_k)
