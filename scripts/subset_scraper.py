from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from PIL import Image
import torch
import numpy as np
from io import BytesIO
import base64

def get_poster_data(url):
    req = Request(
        url=url,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    webpage = urlopen(req).read()#.decode('utf-8')
    sp=soup(webpage,'html.parser')
    imgs=sp.find_all('img')
    titles=sp.find_all('strong')

    if len(imgs)==1:
        if len(titles)==1:
            blob_string=imgs[0]['src'].split(',')[1]
            img=Image.open(BytesIO(base64.b64decode(blob_string)))
            img=torch.tensor(np.array(img))
            title=titles[0].text.strip('\n')
            return img,title
        else:
            raise Exception('more than one titles found.')
    else:
        raise Exception('more than one images found.')

if __name__=='__main__':
    url='https://www.laka.org/docu/affiches/?location=Luxembourg&suchwort=&id=2708'
    img,title=get_poster_data(url)
    print(img.size())
    print(title)
