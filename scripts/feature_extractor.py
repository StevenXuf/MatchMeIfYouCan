import math
import torch
import sys
from multilingual_clip import pt_multilingual_clip
import transformers
from torchvision.transforms import v2

sys.path.append('/home/fxu/alignment/BLIP_Official')

from models.blip_itm import blip_itm

from transformers import AutoProcessor,AutoModel, Blip2Processor, Blip2ForConditionalGeneration

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
def extract_feat_mclip(imgs,txts,device=0):
    clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    size = 1000
    img_features = []
    for i in range(math.ceil(len(imgs)/size)):
        img=imgs[i*size:(i+1)*size]
        img_inputs=clip_processor(images=img,return_tensors='pt').to(device)
        img_features.append(clip.get_image_features(**img_inputs))

    img_features = torch.cat(img_features, dim=0)

    mclip_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
    mclip = pt_multilingual_clip.MultilingualCLIP.from_pretrained(mclip_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(mclip_name)
    txt_features = []
    for j in range(math.ceil(len(txts)/size)):
        txt=txts[j*size:(j+1)*size]
        txt_features.append(mclip.forward(txt,tokenizer=tokenizer))

    txt_features = torch.cat(txt_features, dim=0)

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

@torch.no_grad
def extract_feat_mblip(imgs,txts,device='cuda'):
    processor = Blip2Processor.from_pretrained("Gregor/mblip-mt0-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Gregor/mblip-mt0-xl", device_map=device, torch_dtype=torch.bfloat16)
    
    image = list(map(lambda img: v2.ToPILImage()(img.permute(2,0,1)), imgs))
    size = 1000
    image_feature = []
    for i in range(math.ceil(len(image)/size)):
        img_inputs = processor(images=image[i*size:(i+1)*size], return_tensors="pt").to(device, torch.bfloat16)
        image_feature.append(model.vision_model(img_inputs["pixel_values"]).last_hidden_state[:,0,:])
    image_feature = torch.cat(image_feature, dim=0)
        
    text_feature = torch.zeros((len(txts), image_feature.size(-1)), device=device, dtype=torch.bfloat16)
    proj = torch.nn.Linear(model.language_model.config.hidden_size, image_feature.size(-1), bias=False).to(device, dtype=torch.bfloat16)
    inputs = processor(text=txts, return_tensors="pt", padding=True, truncation=True).to(device,torch.bfloat16)
    for i in range(len(txts)):
        text_feature[i,:] = proj(model.language_model.decoder.embed_tokens(inputs["input_ids"][i])[0,:])

    return image_feature, text_feature
