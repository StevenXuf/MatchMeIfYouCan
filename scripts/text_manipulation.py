import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import spacy
import numpy as np
import logging

from datasets import Dataset
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud
from transformers import AutoModel,AutoTokenizer
from termcolor import colored

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import config
from clean_text_dataset import clean_text,get_subfix,remove_sprcial_chars,store_dataset

def get_contents(topics,role,model,is_meta=False,path='../data/impresso'):
    article_dfs=[]
    article_related=[]
    article_unrelated=[]
    title_related=[]
    title_unrelated=[]
    
    for topic in topics:
        print(topic)
        file_name='_'.join(topic.lower().split(' '))
        df_related=pd.read_csv(path+f'/{file_name}_related.csv',sep=';',header=0)
        df_unrelated=pd.read_csv(path+f'/{file_name}_unrelated.csv',sep=';',header=0)
        
        df_related=df_related.drop(columns=['title.1','content.1'],inplace=False)
        df_unrelated=df_unrelated.drop(columns=['title.1','content.1'],inplace=False)
        article_dfs.append(df_related)
        article_dfs.append(df_unrelated)
       
        content_related=remove_sprcial_chars(df_related['content']).tolist()
        content_unrelated=remove_sprcial_chars(df_unrelated['content']).tolist()
        article_related.append(content_related)
        article_unrelated.append(content_unrelated)
        
        title_re=remove_sprcial_chars(df_related['title']).tolist()
        title_unre=remove_sprcial_chars(df_unrelated['title']).tolist()
        title_related.append(title_re)
        title_unrelated.append(title_unre)
   
    if is_meta:
        subfix='meta'
    else:
        subfix=get_subfix(role)

        corpus=article_related+article_unrelated+title_related+title_unrelated

        corpus_list=denest(corpus)
        results=clean_text(corpus_list,role,model,seed=42,n_gpu=1,batch_size=256,path=path+'/subset')
    
        re_ordered_results=re_order(corpus,results['clean_data'])
        print(len(corpus))
        print(len(re_ordered_results))
        idx=len(corpus)//4
        article_related,article_unrelated,title_related,title_unrelated=re_ordered_results[:idx],re_ordered_results[idx:2*idx],re_ordered_results[2*idx:3*idx],re_ordered_results[3*idx:]

    article_word_cnt=[[len(word_tokenize(content)) for content in cont_lst] for cont_lst in article_related+article_unrelated]
    title_word_cnt=[[len(word_tokenize(content)) for content in cont_lst] for cont_lst in title_related+title_unrelated]

    plot_article_distribution(topics,article_related,article_unrelated)
    plot_wordcloud(topics,lemmatize(article_related),lemmatize(article_unrelated),fig_name=f'articles_{subfix}',exclude_topics=False)
    plot_wordcloud(topics,lemmatize(title_related),lemmatize(title_unrelated),fig_name=f'titles_{subfix}',exclude_topics=False)
    plot_word_distribution(topics,article_word_cnt,fig_name=f'articles')
    plot_word_distribution(topics,title_word_cnt,fig_name=f'titles')
    
    return article_dfs,article_related+article_unrelated,title_related+title_unrelated

def length_count(text):
    if len(text.strip(' '))==0:
        return 0
    else:
        return len(text.strip(' ').split(' '))

def plot_article_distribution(topics,article_related,article_unrelated,fontsize=18):
    barWidth = 0.25
    n_topics=len(topics)
    fig,ax=plt.subplots(figsize=(12,8)) 

    cnt_articles_related=list(map(len,article_related))
    cnt_articles_unrelated=list(map(len,article_unrelated))
    
    br1 = np.arange(n_topics) 
    br2 = [x + barWidth for x in br1] 

    bar_related=ax.bar(br1,cnt_articles_related, color ='orange', width = barWidth,edgecolor='black',label ='Related') 
    bar_unrelated=ax.bar(br2,cnt_articles_unrelated, color='tomato', width = barWidth,edgecolor='black',label ='Unrelated')
    
    for i in range(len(bar_related)):
        bar_re=bar_related[i]
        bar_unre=bar_unrelated[i]
        val_re = bar_re.get_height()
        val_unre=bar_unre.get_height()
        ax.text(
            bar_re.get_x() + bar_re.get_width() / 2,  # x position of the text
            val_re+ 0.15,                         # y position slightly above the bar
            f"{val_re}",                          # label text (the height of the bar)
            ha="center",                        # center horizontally
            va="bottom",                        # align text at the bottom
            fontweight="bold",                   # make text bold (optional)
            fontsize=fontsize
        )
        ax.text(
            bar_unre.get_x() + bar_unre.get_width() / 2,  # x position of the text
            val_unre+ 0.15,                         # y position slightly above the bar
            f"{val_unre}",                          # label text (the height of the bar)
            ha="center",                        # center horizontally
            va="bottom",                        # align text at the bottom
            fontweight="bold",                   # make text bold (optional)
            fontsize=fontsize
            )

    ax.grid(color ='grey',
        linestyle ='-', linewidth = 0.5,
        alpha = 0.2)

    ax.set_ylabel('Number of articles', fontweight ='bold', fontsize = fontsize) 
    ax.set_xticks([r + barWidth/2 for r in range(len(cnt_articles_related))],topics,fontweight ='bold',fontsize=fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc='upper center',prop={'weight':'bold','size':fontsize})
    plt.tight_layout()
    plt.savefig('../figures/article_barplot.pdf')

def plot_word_distribution(topics,word_count,fig_name='articles',fontsize=18):
    n_topics=len(topics)
    fig,ax=plt.subplots(figsize=(12,8))
    colors = ['peachpuff', 'orange', 'tomato']

    boxplt1=ax.boxplot([word_count[i] for i in range(len(word_count)//2)],positions=range(1,n_topics*3+1,3),patch_artist=True,boxprops=dict(color='black',facecolor=colors[1]))
    boxplt2=ax.boxplot([word_count[i] for i in range(len(word_count)//2,len(word_count))],positions=range(2,n_topics*3+2,3),patch_artist=True,boxprops=dict(color='black',facecolor=colors[2])) 

    for median in boxplt1['medians']+boxplt2['medians']:
        median.set(color='black', linewidth=2)

    legend1 = plt.Line2D([0], [0], color=colors[1], lw=4, label='Related')
    legend2 = plt.Line2D([0], [0], color=colors[2], lw=4, label='Unrelated')
    ax.legend(handles=[legend1, legend2], loc='upper center',prop={'weight':'bold','size':fontsize})

    ax.set_xticks(np.arange(1.5,n_topics*3+1.5,3),topics,fontsize=fontsize,fontweight='bold')
    ax.set_ylabel(f"Number of words for {fig_name}",fontsize=fontsize,fontweight='bold')
    
    ax.grid(color ='grey',
        linestyle ='-', linewidth = 0.5,
        alpha = 0.2)
    
    plt.tight_layout()
    plt.savefig(f"../figures/word_boxplot_{fig_name}.pdf")

def plot_wordcloud(topics,article_related,article_unrelated,fig_name='articles',exclude_topics=False):
    n_topics=len(topics)
    fig,axes=plt.subplots(1,n_topics,figsize=(n_topics*4,3))
    
    extra_stopwords=config.extra_stopwords
    stopwords_used = stopwords.words('german')
    if exclude_topics:
        stopwords_used = set(stopwords_used+extra_stopwords)
    
    for i in range(n_topics):
        all_texts=word_tokenize('\n'.join(article_related[i]+article_unrelated[i]),language='german')
        all_texts=' '.join(all_texts)

        wordcloud = WordCloud(width=800, height=400, background_color='white',
                      stopwords=stopwords_used).generate(all_texts)
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(topics[i],fontsize=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/wordcloud_{fig_name}.pdf')

def re_order(articles,results):
    divided_list = []
    index = 0
    cnts=list(map(len,articles))

    for length in cnts:
        sublist = results[index:index + length]
        divided_list.append(sublist)
        index += length

    return divided_list

def translate(model_id,articles):
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id)

    denested_articles=denest(articles)
    
    results=[]
    for article in tqdm(denested_articles):
        inputs = tokenizer(article, return_tensors="pt")
        generated_seq=model.generate(**inputs)
        translated_text = tokenizer.decode(generated_seq[0], skip_special_tokens=True)
        results.append(translated_text)
    
    re_ordered_results=re_order(articles,results)

    return re_ordered_results


def lemmatize(articles):
    nlp=spacy.load('de_core_news_sm')
    
    denested_articles=denest(articles)

    results=[]
    for article in tqdm(denested_articles):
        doc = nlp(article)
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha]
        lemmatized_text = ' '.join(lemmatized_words)
        results.append(lemmatized_text)

    re_ordered_results=re_order(articles,results)

    return re_ordered_results

def denest(nested_list):
    return [item for sublist in nested_list for item in sublist]

if __name__=='__main__':
    torch.manual_seed(0)
    topics=config.english_topics
    role=config.system_role_editor
    model=config.qwen
    logging.basicConfig(level=logging.DEBUG)
    _,articles,titles=get_contents(topics,role,model,is_meta=True)
