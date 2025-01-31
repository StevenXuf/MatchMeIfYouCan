import pandas as pd
from subset_scraper import get_poster_data
import torch


def load_data(in_file='poster_article_pairs.csv',out_file='posters_with_texts.pt'):
    web_info=pd.read_csv(f'./{in_file}',sep=',',header=0)
    res={'images':[],'texts':[]}
    cnt=0
    for poster_address in web_info['Poster'].tolist():
        cnt+=1
        img,text=get_poster_data(poster_address)
        res['images'].append(img)
        res['texts'].append(text)
    print('Num of posters: ',cnt)
    torch.save(res,f'./{out_file}')
    return res

if __name__=='__main__':
    poster_data=load_data()
    print(poster_data['texts'])
