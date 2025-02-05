import pandas as pd
import transformers
from datasets import Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import spacy
import numpy as np

from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud,STOPWORDS
from transformers import AutoModel,AutoTokenizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import config

def get_contents(topics,dtype='meta'):
    article_dfs=[]
    article_related=[]
    article_unrelated=[]
    title_related=[]
    title_unrelated=[]
    path='../data/impresso'
    
    for topic in topics:
        print(topic)
        file_name='_'.join(topic.lower().split(' '))
        df_related=pd.read_csv(path+f'/{file_name}_related.csv',sep=';',header=0)
        df_unrelated=pd.read_csv(path+f'/{file_name}_unrelated.csv',sep=';',header=0)
        
        df_related=df_related.drop(columns=['title.1','content.1'],inplace=False)
        df_unrelated=df_unrelated.drop(columns=['title.1','content.1'],inplace=False)
        article_dfs.append(df_related)
        article_dfs.append(df_unrelated)
       
        content_related=df_related['content'].fillna('').tolist()
        content_unrelated=df_unrelated['content'].fillna('').tolist()
        article_related.append(content_related)
        article_unrelated.append(content_unrelated)
        
        title_re=df_related['title'].fillna('').tolist()
        title_unre=df_unrelated['title'].fillna('').tolist()
        title_related.append(title_re)
        title_unrelated.append(title_unre)
    
    if dtype=='clean':
        corpus=article_related+article_unrelated+title_related+title_unrelated
        corpus=manipulate_texts(
            config.llama,
            config.system_role_editor,
            corpus)
        idx=len(corpus)//4
        article_related,article_unrelated,title_related,title_unrelated=corpus[:idx],corpus[idx:2*idx],corpus[2*idx:3*idx],corpus[3*idx:]

    article_word_cnt=[list(map(length_count,cont_lst)) for cont_lst in article_related+article_unrelated]
    title_word_cnt=[list(map(length_count,cont_lst)) for cont_lst in title_related+title_unrelated]

    plot_article_distribution(topics,article_related,article_unrelated)
    plot_wordcloud(topics,article_related,article_unrelated,fig_name=f'articles_{dtype}')
    plot_wordcloud(topics,title_related,title_unrelated,fig_name=f'titles_{dtype}')
    plot_word_distribution(topics,article_word_cnt,fig_name=f'articles_{dtype}')
    plot_word_distribution(topics,title_word_cnt,fig_name=f'titles_{dtype}')
    
    return article_dfs,article_related+article_unrelated,title_related+title_unrelated

def length_count(text):
    if len(text.strip(' '))==0:
        return 0
    else:
        return len(text.strip(' ').split(' '))

def plot_article_distribution(topics,article_related,article_unrelated):
    barWidth = 0.25
    n_topics=len(topics)
    fig,ax=plt.subplots(figsize=(12,8)) 

    cnt_articles_related=list(map(len,article_related))
    cnt_articles_unrelated=list(map(len,article_unrelated))
    
    br1 = np.arange(n_topics) 
    br2 = [x + barWidth for x in br1] 

    bar_related=ax.bar(br1,cnt_articles_related, color ='#1f78b4', width = barWidth, edgecolor ='grey', label ='related',alpha=1) 
    bar_unrelated=ax.bar(br2,cnt_articles_unrelated, color='#ff7f00', width = barWidth,edgecolor ='grey', label ='unrelated',alpha=1)
    
    for i in range(len(bar_related)):
        bar_re=bar_related[i]
        bar_unre=bar_unrelated[i]
        val_re = bar_re.get_height()
        val_unre=bar_unre.get_height()
        ax.text(
            bar_re.get_x() + bar_re.get_width() / 2,  # x position of the text
            val_re+ 0.5,                         # y position slightly above the bar
            f"{val_re}",                          # label text (the height of the bar)
            ha="center",                        # center horizontally
            va="bottom",                        # align text at the bottom
            fontweight="bold"                   # make text bold (optional)
        )
        ax.text(
            bar_unre.get_x() + bar_unre.get_width() / 2,  # x position of the text
            val_unre+ 0.5,                         # y position slightly above the bar
            f"{val_unre}",                          # label text (the height of the bar)
            ha="center",                        # center horizontally
            va="bottom",                        # align text at the bottom
            fontweight="bold"                   # make text bold (optional)
        )

    ax.grid(color ='grey',
        linestyle ='-', linewidth = 0.5,
        alpha = 0.2)

    ax.set_ylabel('Number of articles', fontweight ='bold', fontsize = 15) 
    ax.set_xticks([r + barWidth/2 for r in range(len(cnt_articles_related))],topics,fontweight ='bold',fontsize=15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc='upper center',prop={'weight':'bold','size':15})
    plt.tight_layout()
    plt.savefig('../figures/article_barplot.pdf')

def plot_word_distribution(topics,word_count,fig_name='articles'):
    n_topics=len(topics)
    fig,ax=plt.subplots(figsize=(12,8))
    colors = ['peachpuff', 'orange', 'tomato']

    boxplt1=ax.boxplot([word_count[i] for i in range(len(word_count)//2)],positions=range(1,n_topics*3+1,3),patch_artist=True,boxprops=dict(color='black',facecolor=colors[1]))
    boxplt2=ax.boxplot([word_count[i] for i in range(len(word_count)//2,len(word_count))],positions=range(2,n_topics*3+2,3),patch_artist=True,boxprops=dict(color='black',facecolor=colors[2])) 
    '''
    for i in range(int(len(boxplt1['boxes'])/2)):
        for patch in boxplt1['boxes'][2*i:2*(i+1)]:
            patch.set(facecolor=colors[i])
    '''

    for median in boxplt1['medians']+boxplt2['medians']:
        median.set(color='black', linewidth=2)

    legend1 = plt.Line2D([0], [0], color=colors[1], lw=4, label='Related')
    legend2 = plt.Line2D([0], [0], color=colors[2], lw=4, label='Unrelated')
    ax.legend(handles=[legend1, legend2], loc='upper center',prop={'weight':'bold','size':15})

    ax.set_xticks(np.arange(1.5,n_topics*3+1.5,3),topics,fontsize=15,fontweight='bold')
    ax.set_ylabel(f"Number of words for {fig_name}",fontsize=15,fontweight='bold')
    
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
        all_texts='\n'.join(article_related[i]+article_unrelated[i])
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

def clean_metadata(batch,pipeline,system_role):
    
    messages = [[
            {"role": "system", "content": system_role},
            {"role": "user", "content": article},
        ] for article in batch["metadata"]]
    
    cleaned_results = pipeline(messages,max_new_tokens=3076)
    batch["cleaned_data"] = [result["generated_text"][-1]['content'] for result in cleaned_results]

    return batch

def manipulate_texts(model_id,system_role,articles):
    results=[]
    denested_articles=[item for sublist in articles for item in sublist]
    dataset=Dataset.from_dict({'metadata':denested_articles})
    
    pipeline= transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    '''
    for article in tqdm(denested_articles):
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": article},
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=3076
        )
        clean_article=outputs[0]["generated_text"][-1]['content']
        results.append(clean_article)
        print(article)
        print('-'*40)
        print(clean_article)
        print('*'*40)
    '''

    clean_dataset=dataset.map(clean_metadata,batched=True,batch_size=16,fn_kwargs={'pipeline':pipeline,'system_role':system_role})
    re_ordered_results=re_order(articles,clean_dataset['cleaned_data'])

    return re_ordered_results


def translate(model_id,articles):
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id)

    denested_articles=[item for sublist in articles for item in sublist]
    
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
    
    denested_articles=[item for sublist in articles for item in sublist]

    results=[]
    for article in tqdm(denested_articles):
        doc = nlp(article)
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha]
        lemmatized_text = ' '.join(lemmatized_words)
        results.append(lemmatized_text)

    re_ordered_results=re_order(articles,results)

    return re_ordered_results

if __name__=='__main__':
    torch.manual_seed(0)
    topics=config.english_topics
    _,articles,titles=get_contents(topics,dtype='clean')
