import math
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from load_poster_data import load_data
import config
from feature_extractor import extract_feat_clip, extract_feat_blip

def get_poster_data(in_file,out_file):
    if os.path.exists(f'{out_file}'):
        swiss_poster_data=torch.load(f'{out_file}',weights_only=True)
    else:
        swiss_poster_data=load_data(in_file,out_file)
    return swiss_poster_data

def get_poster_subset(in_file,out_file,anno_file):
    poster_annotation=pd.read_excel(f'{anno_file}')
    poster_annotation=poster_annotation.fillna(0)
    poster_annotation[['children','environmentalism']]=poster_annotation[['children','environmentalism']].astype('int64')

    df=poster_annotation.drop(['image','text','doom','children'],axis=1)
    is_all_zeros=df.eq(0).all(axis=1)
    df=df[~is_all_zeros]

    poster_indices=df.index.tolist()
    
    clean_images=[]
    clean_texts=[]
    swiss_poster_data=get_poster_data(in_file,out_file)
    for idx in poster_indices:
        clean_images.append(swiss_poster_data['images'][idx])
        clean_texts.append(swiss_poster_data['texts'][idx])
    return {'images':clean_images,'texts':clean_texts,'ids':poster_indices,'anno':df}

def img_transform():
    transform1 = v2.Compose([
    lambda img: img[:,:,:3] if img.size(2)==4 else img,
    lambda img: torch.cat((img,img,img),dim=0) if img.size(2)==1 else img,
    lambda img: img/256 if img.dtype!=torch.float else img,
    lambda img: img.permute(2,0,1),
    v2.Resize(size=(224, 224), antialias=True)
])
    transform2 = v2.Compose([
    lambda img: img[:,:,:3] if img.size(2)==4 else img,
    lambda img: torch.cat((img,img,img),dim=0) if img.size(2)==1 else img
])
    transform3 = v2.Compose([
        lambda img: img[:,:,:3] if img.size(2)==4 else img,
        lambda img: torch.cat((img,img,img),dim=0) if img.size(2)==1 else img,
        lambda img: img.permute(2,0,1),
        v2.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        lambda img: img/256 if img.dtype!=torch.float else img,
        v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        lambda img: img.permute(1,2,0)
        ])
    return transform1,transform2,transform3

def plot_poster_dist(dates,cnts):
    fig, ax = plt.subplots(figsize=(7,3))

    bars=ax.bar(dates,cnts,color=colors)
    ax.set_xticklabels(dates, fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', 
                ha='center', va='bottom', fontsize=6)

    ax.set_ylabel('frequency')

    plt.savefig('../figures/poster_dist.pdf')

def text_process(title,line_length=4):
    words=title.strip(' ').split(' ')
    res=''
    for i in range(len(words)):
        res+=words[i]
        if (i+1)%line_length==0:
            res+='\n'
        else:
            res+=' '
    res=res.strip(' ').strip('\n')
    return res

def plot_poster_with_title(imgs,titles,fig_name,ids):
    total=len(imgs)
    n_col=5
    n_row=int(math.ceil(total/n_col))
    subplt_size=3

    fig,axes=plt.subplots(n_row,n_col,figsize=(n_col*subplt_size,n_row*subplt_size))
    transform1,_,_=img_transform()
    for ii in range(n_row*n_col):
        i,j=ii//n_col,ii%n_col
        if ii<total: 
            img=imgs[ii]
            img=transform1(img)
            img=img.permute(1,2,0)
            axes[i,j].imshow(img)
            axes[i,j].set_title(f"{text_process(titles[ii].split(';')[0])}",fontsize=8)
            if ii in ids:
                for spine in axes[i,j].spines.values():
                    spine.set_color('green')
                    spine.set_linewidth(5)
            else:
                for spine in axes[i,j].spines.values():
                    spine.set_visible(False)

            axes[i,j].set_xticks([])  # Hide x-axis ticks
            axes[i,j].set_yticks([])

    plt.subplots_adjust(left=0.0, right=1.0, top=.97, bottom=0.02, wspace=0.0, hspace=0.31)
    plt.savefig(f'../figures/{fig_name}_posters.pdf')


def extract_features(data,model):
    features={}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txts=data['texts']
    _,transform2,transform3=img_transform()
    print('*'*10+model+'*'*10)

    if model.lower() == 'clip':
        imgs=list(map(transform2,data['images']))
        features['image features'],features['text features']=extract_feat_clip(imgs,txts,device)
    elif model.lower() == 'blip':
        imgs=list(map(transform3,data['images']))
        features['image features'],features['text features']=extract_feat_blip(imgs,txts,device)
    else:
        raise Exception('No such a model.')
    return features

def plot_cosine(features,model):
    cos_sim=pairwise_cosine_similarity(features['image features'].cpu(),features['text features'].cpu())
    plt.figure(figsize=(3,3))
    plt.imshow(cos_sim)
    plt.title('Img-txt Cosine Similarity')
    plt.savefig(f'../figures/cosine_{model}.pdf')

def plot_topic_img_sim(topics,images,model_name):
    features=extract_features({'images':images,'texts':topics},model_name)
    topic_image_scores=pairwise_cosine_similarity(features['text features'],features['image features']).cpu()
    plt.figure(figsize=(18,4))
    plt.imshow(topic_image_scores)
    plt.yticks(range(len(topics)), topics, fontsize=12)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image,extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(topic_image_scores.shape[1]):
        for y in range(topic_image_scores.shape[0]):
            plt.text(x, y, f"{topic_image_scores[y, x]:.2f}", ha="center", va="center", size=8)
    
    for side in ["left", "top", "right", "bottom"]:
      plt.gca().spines[side].set_visible(False)
    
    plt.xlim([-0.5, len(images) - 0.5])
    plt.ylim([len(topics) + 0.5, -2])
    
    plt.tight_layout()
    plt.savefig(f'../figures/topic2img_sim_{model_name}.pdf')


if __name__=='__main__':
    model_name='clip'
    topics=config.english_topics
    in_file,out_file,anno_file=config.in_file,config.out_file,config.anno_file

    poster=get_poster_data(in_file,out_file)
    subset=get_poster_subset(in_file,out_file,anno_file)

    plot_poster_with_title(poster['images'],poster['texts'],'Laka',subset['ids'])
    plot_topic_img_sim(topics,subset['images'],model_name)
