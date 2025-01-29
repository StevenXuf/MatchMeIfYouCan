import torch
import sys
sys.path.append('/home/users/u101139/alignment/BLIP_Official')

from models.blip import blip_feature_extractor
from models.blip_itm import blip_itm

from transformers import AutoProcessor,AutoModel
from tqdm import tqdm

@torch.no_grad
def extract_feat_clip(imgs,txts,device=0):
    model = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    txt_inputs=processor(text=txts,return_tensors='pt',padding=True,truncation=True).to(device)
    img_inputs=processor(images=imgs,return_tensors='pt').to(device)
    txt_features=model.get_text_features(**txt_inputs)
    img_features=model.get_image_features(**img_inputs)

    return img_features,txt_features

@torch.no_grad
def extract_feat_blip(imgs,txts,device=0):
    n_dim_features=256
    model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=224, vit='base').eval()
    model=model.to(device)
    
    img_features=torch.zeros(len(imgs),n_dim_features).to(device)
    txt_features=torch.zeros(len(txts),n_dim_features).to(device)

    for i in range(len(imgs)):
        image=imgs[i].permute(2,0,1).unsqueeze(0).to(device)
        caption=['']
        image_feature,text_feature = model(image, caption, match_head='itc')
        img_features[i,:]=image_feature.detach()
    
    for j in range(len(txts)):
        dummy_image=torch.randn(1,3,224,224).to(device)
        caption=txts[j]
        image_feature,text_feature = model(dummy_image, caption, match_head='itc')
        txt_features[j,:]=text_feature.detach()

    return img_features,txt_features
