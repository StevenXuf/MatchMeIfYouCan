import pandas as pd
import transformers
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import spacy

from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud,STOPWORDS
from transformers import AutoModel,AutoTokenizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import config

def get_contents(topics):
    article_dfs=[]
    article_related=[]
    article_unrelated=[]
    title_related=[]
    title_unrelated=[]
    path='../data/impresso'
    article_word_cnt=[]
    title_word_cnt=[]
    
    for topic in topics:
        print(topic)
        if topic=='environmentalism':
            df_related=pd.read_csv(path+'/environ_related.csv',sep=';',header=0)
            df_unrelated=pd.read_csv(path+'/environ_unrelated.csv',sep=';',header=0)
        elif topic=='nuclear accidents':
            df_related=pd.read_csv(path+'/accidents_related.csv',sep=';',header=0)
            df_unrelated=pd.read_csv(path+'/accidents_unrelated.csv',sep=';',header=0)
        elif topic=='nuclear waste':
            df_related=pd.read_csv(path+'/waste_related.csv',sep=';',header=0)
            df_unrelated=pd.read_csv(path+'/waste_unrelated.csv',sep=';',header=0)
        elif topic=='protest':
            df_related=pd.read_csv(path+'/protest_related.csv',sep=';',header=0)
            df_unrelated=pd.read_csv(path+'/protest_unrelated.csv',sep=';',header=0)
        else:
            raise Exception('No such topic!')
        df_related=df_related.drop(columns=['title.1','content.1'],inplace=False)
        df_unrelated=df_unrelated.drop(columns=['title.1','content.1'],inplace=False)
        article_dfs.append(df_related)
        article_dfs.append(df_unrelated)
       
        content_related=df_related['content'].tolist()
        content_unrelated=df_unrelated['content'].tolist()
        article_related.append(content_related)
        article_unrelated.append(content_unrelated)
        article_word_cnt.append(list(map(length_count,content_related)))
        article_word_cnt.append(list(map(length_count,content_unrelated)))
        
        title_re=df_related['title'].tolist()
        title_unre=df_unrelated['title'].tolist()
        title_related.append(title_re)
        title_unrelated.append(title_unre)
        title_word_cnt.append(list(map(length_count,title_re)))
        title_word_cnt.append(list(map(length_count,title_unre)))

    plot_article_distribution(topics,article_related,article_unrelated)
    plot_wordcloud(topics,article_related,article_unrelated)
    plot_word_distribution(topics,article_word_cnt,fig_name='articles')
    plot_word_distribution(topics,title_word_cnt,fig_name='titles')

    return article_dfs,article_related+article_unrelated,title_related+title_unrelated

def length_count(text):
    return len(text.strip(' ').split(' '))

def plot_article_distribution(topics,article_related,article_unrelated):
    barWidth = 0.25
    n_topics=len(topics)
    fig,ax=plt.subplots(figsize=(12,8)) 

    cnt_articles_related=list(map(len,article_related))
    cnt_articles_unrelated=list(map(len,article_unrelated))
    
    br1 = np.arange(n_topics) 
    br2 = [x + barWidth for x in br1] 

    bar_related=ax.bar(br1,cnt_articles_related, color ='#D39200', width = barWidth, edgecolor ='grey', label ='related',alpha=1) 
    bar_unrelated=ax.bar(br2,cnt_articles_unrelated, color='#0072B2', width = barWidth,edgecolor ='grey', label ='unrelated',alpha=1)
    
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
    plt.savefig('../figures/article_barplot.pdf')

def plot_word_distribution(topics,word_count,fig_name='articles'):
    n_topics=len(topics)
    fig,ax=plt.subplots(figsize=(12,8))
    colors = ['peachpuff', 'orange', 'tomato']

    boxplt1=ax.boxplot([word_count[i] for i in range(len(word_count)) if i%2==0],positions=range(1,n_topics*3+1,3),patch_artist=True,boxprops=dict(color='black',facecolor=colors[1]))
    boxplt2=ax.boxplot([word_count[i] for i in range(len(word_count)) if i%2==1],positions=range(2,n_topics*3+2,3),patch_artist=True,boxprops=dict(color='black',facecolor=colors[2])) 
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

    ax.set_xticks([1.5,4.5,7.5],topics,fontsize=15,fontweight='bold')
    ax.set_ylabel(f"Number of words for {fig_name}",fontsize=15,fontweight='bold')
    
    ax.grid(color ='grey',
        linestyle ='-', linewidth = 0.5,
        alpha = 0.2)

    plt.savefig(f"../figures/word_boxplot_{fig_name}.pdf")

def plot_wordcloud(topics,article_related,article_unrelated,fig_name='article',exclude_topics=False):
    n_topics=len(topics)
    fig,axes=plt.subplots(1,n_topics,figsize=(n_topics*4,3))
    
    extra_stopwords=['Umweltschutz', 'Waldsterben','Unfall','Abfall','Unfälle','Abfälle','Unfällen','Protest']
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

def manipulate_texts(model_id,system_role,articles):
    results=[]
    denested_articles=[item for sublist in articles for item in sublist]

    pipeline= transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    for article in tqdm(denested_articles):
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": article},
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=3076
        )
        results.append(outputs[0]["generated_text"][-1]['content'])
    
    re_ordered_results=re_order(articles,results)

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
    _,articles,titles=get_contents(config.english_topics)
    corpus=articles
    
    '''
    corpus=manipulate_texts(
        config.llama_3_1_8b_instruct,
        config.system_role_editor,
        corpus)
    idx=len(corpus)//2 
    plot_wordcloud(config.english_topics,
            corpus[:idx],
            corpus[idx:],
            fig_name='articles_ex_de')
    '''
