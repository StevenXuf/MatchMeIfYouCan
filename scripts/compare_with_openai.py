import base64
import torch
import torchvision.transforms as transforms
from io import BytesIO
from openai import OpenAI

import config

from read_posters import get_file_path
from text_manipulation import get_contents
from poster_manipulation import get_poster_subset,get_precision_recall,img_transform
from cross_modal_retrieval import get_targets

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # Convert to JPEG/PNG
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_img_paths(path='/project/home/p200630/Data/deduped'):
    paths=get_file_path(path)
    img_paths=[]
    for country in list(paths.keys()):
        img_paths.extend(paths[country]['img_path'])
    
    return img_paths

def get_img2txt_results(image,text_list,api_key):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful asistant that solves problems and gives answers in a concise,accurate, and brief way."},
            {"role": "user", "content": [
                {"type": "text", "text": f"I want you to judge the similarity of an given image with respect to a list of textual descriptions. You should return the index order of the 10 most similar decsriptions where the most similar text to the image shall be in the first position and the most dissimilar one shall be in the last. Use json format for your answer. Here is the text list: {text_list}."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(image)}"}}
            ]}
        ],
        max_tokens=50,
        temperature=0.2,  # More focused response
        top_p=0.5
        )
    reply=response.choices[0].message.content
    return reply

def get_txt2img_results(images,text,api_key):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful asistant that solves problems and gives answers in a concise,accurate, and brief way."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"I want you to judge the similarity of an given textual description with respect to a list of images. You should return the index order of the 10 most similar images where the most similar image to the text shall be in the first position and the most dissimilar one shall be in the last. Use json format for your answer. Remember there are only 22 images so the index starting from 0 shall not be larger than 21. Here is the text: {text}."}]+[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in images]}
            ],
            max_tokens=50,
            temperature=0.2,  # More focused response
            top_p=0.5
            )
    
    reply=response.choices[0].message.content
    
    return reply

def convert_idx_to_score(idx,size):
    cosine=torch.zeros(size)
    for i,d in enumerate(idx):
        score=torch.linspace(1,0.1,steps=len(d))
        cosine[i,d]=score
    return cosine

if __name__=='__main__':
    n_samples=5
    _,articles,titles=get_contents(config.english_topics,config.system_role_summarizer,config.qwen,is_meta=False)
    
    transform=transforms.ToPILImage()
    _,transform2,transform3=img_transform()
    poster=get_poster_subset(config.in_file,config.out_file,config.anno_file)
    images=list(map(lambda x: transform(transform2(x).permute(2,0,1)),poster['images']))
    
    for task in ['img2txt','txt2img']:
        targets=get_targets(poster,articles,task)
        if task=='img2txt':
            preds=convert_idx_to_score(config.img2txt_top10,targets.size())
        else:
            preds=convert_idx_to_score(config.txt2img_top10,targets.size())
        
        for k in [1,5,10]:
            get_precision_recall(preds,targets,k)
    
    '''
    #img2txt
    preds=[]
    for i in range(len(images)):
        reply=get_img2txt_results(images[i],denest(articles),config.api_key)
        print(f'Predictions for images {i}:')
        print(reply)
    
    #txt2img
    images = list(map(pil_to_base64,images))
    for j in [45,59]:
        reply=get_txt2img_results(images,denest(articles)[j],config.api_key)
        print(f'Predictions for text {j}:')
        print(reply)
    '''
