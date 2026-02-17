from PIL import Image
import torchvision.transforms as transforms

import torch
import os
import sys

from poster_manipulation import plot_poster_dist

def get_file_path(path):
    poster_paths={}
    
    dirs=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dr in dirs:
        current_path=os.path.join(path,dr)
        txt_paths=[os.path.join(current_path,tf) for tf in os.listdir(current_path) if os.path.isfile(os.path.join(current_path,tf)) and tf.endswith('.txt')]
        img_paths=[os.path.join(current_path,tf) for tf in os.listdir(current_path) if os.path.isfile(os.path.join(current_path,tf)) and tf.endswith('.png')]
        assert len(txt_paths)==len(img_paths),"Number of files not match!"

        poster_paths[dr]={'txt_path':sorted(txt_paths),'img_path':sorted(img_paths)}

    return poster_paths


def get_poster_for_country(country,img_store_path='/data/data_fxu/deduped'):
    country=country.capitalize()
    paths=get_file_path(img_store_path)

    if country in paths:
        country_dict=paths[country]
        transform = transforms.ToTensor()
        imgs=[]
        caps=[]
        for i in range(len(country_dict['img_path'])):
            if country_dict['txt_path'][i].split('.')[0]==country_dict['img_path'][i].split('.')[0]:
                imgs.append(transform(Image.open(country_dict['img_path'][i]).convert('RGB').resize((224,224))))
                with open(country_dict['txt_path'][i], "r") as file:
                    content=file.read()
                    caps.append(content)
    else:
        raise Exception('No such a country. Retry.')
        sys.exit(1)
    return imgs,caps

def get_all_poster(img_store_path='/data/data_fxu/deduped'):
    paths=get_file_path(img_store_path)
    images=[]
    captions=[]
    transform=transforms.ToTensor()
    for country in list(paths.keys()):
        country_dict=paths[country]
        for i in range(len(country_dict['img_path'])):
            if country_dict['txt_path'][i].split('.')[0]==country_dict['img_path'][i].split('.')[0]:
                images.append(transform(Image.open(country_dict['img_path'][i]).convert('RGB').resize((224,224))))
                with open(country_dict['txt_path'][i], "r") as file:
                    content=file.read()
                    captions.append(content)
    return images,captions

if __name__=='__main__':
    country='switzerland'
    #images,captions=get_poster_for_country(country)
    #plot_poster_dist(captions)
    images,captions=get_all_poster()
    print(len(images))
