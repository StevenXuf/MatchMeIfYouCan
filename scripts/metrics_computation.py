from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.classification import MulticlassPrecision,MulticlassRecall,MultilabelPrecision,MultilabelRecall

import torch
import numpy as np

import config
from text_manipulation import get_contents,manipulate_texts,translate,lemmatize
from feature_extractor import extract_feat_clip,extract_feat_blip

def compute_metrics_via_classic_methods(method,query,corpus,top_k=10):
    if method=='bow':
        method=CountVectorizer()
    elif method=='tfidf':
        method=TfidfVectorizer()
    cnt_related=list(map(len,corpus))
    corpus=[item for sublist in corpus for item in sublist]
    X = method.fit_transform(corpus)
    document_term_matrix = X.toarray()
    
    query_vector = method.transform(query).toarray()
    
    similarity_scores = cosine_similarity(query_vector, document_term_matrix)
    recall_accum=0.0
    precision_accum=.0

    for i in range(len(query)):
        start=sum(cnt_related[:i])
        end=sum(cnt_related[:i+1])
        print(start,end)
        curr_range=range(1,document_term_matrix.shape[0])[start:end]
        ranked_indices = np.argsort(similarity_scores[i])[::-1]  # Sort indices in descending order
        
        cnt=0
        print('current range:',curr_range)
        for idx in ranked_indices[:top_k]:
            print(f"Article {idx + 1} with score {similarity_scores[i][idx]:.4f}")
            if idx+1 in curr_range:
                cnt+=1
        recall=cnt/len(curr_range)
        precision=cnt/top_k
        print(f"Recall: {recall*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        recall_accum+=recall
        precision_accum+=precision

    print(f"Avg recall: {recall_accum/len(query)*100:.2f}%")
    print(f"Avg precision: {precision_accum/len(query)*100:.2f}%")


def compute_metrics_via_features(method,query,corpus,top_k=10):
    dummy_img=torch.zeros(224,224,3)
    cnt_related=list(map(len,corpus))
    corpus=[item for sublist in corpus for item in sublist]

    if method=='clip':
        _,query_features=extract_feat_clip([dummy_img]*len(query),query)
        _,corpus_features=extract_feat_clip([dummy_img]*len(corpus),corpus)
    elif method=='blip':
        _,query_features=extract_feat_blip([dummy_img]*len(query),query)
        _,corpus_features=extract_feat_blip([dummy_img]*len(corpus),corpus)
    similarity_scores=pairwise_cosine_similarity(query_features,corpus_features)
    vals,ids=torch.topk(similarity_scores,k=top_k)
    
    recall_accum=.0
    precision_accum=.0

    for i in range(len(query)):
        start=sum(cnt_related[:i])
        end=sum(cnt_related[:i+1])
        print(start,end)
        curr_range=range(1,corpus_features.size(0))[start:end]

        cnt=0
        print('current range:',curr_range)
        for j in range(top_k):
            print(f"Article {ids[i][j].item()+1} with score {vals[i][j].item():.4f}")
            if ids[i][j].item()+1 in curr_range:
                cnt+=1
        recall=cnt/len(curr_range)
        precision=cnt/top_k
        print(f"Recall: {recall*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        recall_accum+=recall
        precision_accum+=precision

    print(f"Avg recall: {recall_accum/len(query)*100:.2f}%")
    print(f"Avg precision: {precision_accum/len(query)*100:.2f}%")


#use features to  ompute precision/recall for de/en full/sum
def compute_results(method,query_type='queries_de',corpus_type='titles',text_transform=False,lemmatization=False):
    _,articles,titles=get_contents(config.english_topics)
    
    if query_type=='queries_de':
        query=config.german_queries
    elif query_type=='queries_en':
        query=config.english_queries
    elif query_type=='prompts_de':
        query=config.german_prompts
    elif query_type=='prompts_en':
        query=config.english_prompts
    else:
        raise Exception("Only 'en' or 'de'")

    if corpus_type=='titles':
        corpus=titles
    elif corpus_type=='articles':
        corpus=articles
    else:
        raise Exception("Only 'titles' or 'articles'")

    if text_transform:
        corpus=manipulate_texts(
            config.llama_3_1_8b_instruct,
            config.system_role_en_summarizer, #change the role as needed.
            corpus)
    
    if lemmatization:
        corpus=lemmatize(corpus)

    if method=='clip' or method=='blip':
        compute_metrics_via_features(method,query,corpus,config.top_k)
    elif method=='bow' or method=='tfidf':
        compute_metrics_via_classic_methods(method,query,corpus,config.top_k)
    else:
        raise Exception('No such a method.')

if __name__=='__main__':
    method='blip'
    query='queries_en'
    corpus='articles'
    text_transform=True
    lemmatization=False
    compute_results(method,query,corpus,text_transform,lemmatization)
