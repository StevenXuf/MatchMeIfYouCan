import laka_config
import matplotlib.pyplot as plt

import os
import hashlib

from PIL import Image
import torch

def get_valid_paths(path):
    subpaths=[]

    for p in os.listdir(path):
        current_path=os.path.join(path,p)
        if os.path.isdir(current_path) and len(os.listdir(current_path))!=0:
            subpaths.append(current_path)
        elif os.path.isdir(current_path) and len(os.listdir(current_path))==0:
            os.rmdir(current_path)

    valid_img_paths=[]
    valid_txt_paths=[]

    for subpath in subpaths:
        current_files=[os.path.join(subpath,f) for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath,f))]
        txt_paths=[path for path in current_files if path.lower().endswith('.txt')]
        img_paths=[path for path in current_files if path not in txt_paths]
        valid_img_paths.extend(img_paths)
        valid_txt_paths.extend(txt_paths)

        assert len(valid_img_paths)==len(valid_txt_paths),"Should have same number of files."

    return valid_img_paths,valid_txt_paths

def compute_image_hash(img_path,algorithm="md5"):
    hash_func = hashlib.new(algorithm)
    with Image.open(img_path) as img:
        img=img.convert("RGB")
        img = img.resize((256, 256))
        img_bytes = img.tobytes()

    hash_func.update(img_bytes)
    return hash_func.hexdigest()

def find_duplicates(valid_paths,dtype='text'):
    contents={}
    duplicates=[]

    if dtype=='text':
        for txt_path in valid_paths:
            with open(txt_path,'r') as txt_file:
                txt=txt_file.read()
                if txt not in contents:
                    contents[txt]=txt_path
                else:
                    duplicates.append(txt_path)
    elif dtype=='image':
        for img_path in valid_paths:
            img_hash=compute_image_hash(img_path)

            if img_hash not in contents:
                contents[img_hash]=img_path
            else:
                duplicates.append(img_path)
    else:
        raise Exception('No such data type.')

    print(f'Number of duplicates for {dtype}: {len(duplicates)}')

    return duplicates

def get_deduped_path(path):
    valid_img_paths,valid_txt_paths=get_valid_paths(path)
    print(f'Total number of images: {len(valid_img_paths)}')

    #txt_duplicates=find_duplicates(valid_txt_paths,dtype='text')
    img_duplicates=find_duplicates(valid_img_paths,dtype='image')
    
    deduplicated_img_paths=[path for path in valid_img_paths if path not in img_duplicates]
    print(f'Total number of images after deduplicating: {len(deduplicated_img_paths)}')

    return deduplicated_img_paths

def create_clean_data(deduped_paths):
    if not os.path.exists(laka_config.deduped_dir):
        os.makedirs(laka_config.deduped_dir)
    
    subfolder_names=[entry.name for entry in os.scandir(laka_config.dir_name) if entry.is_dir()]
    for subfolder in subfolder_names:
        os.makedirs(os.path.join(laka_config.deduped_dir,subfolder),exist_ok=True)

    for i,path in enumerate(deduped_paths):
        file_name=str(i)
        current_folder=os.path.join(laka_config.deduped_dir,path.split('/')[-2])
        img=Image.open(path).convert("RGB")
        _=img.save(os.path.join(current_folder,f'{file_name}.png'))

        txt_path,extension=os.path.splitext(path)
        txt_path+='.txt'
        with open(txt_path,'r') as txt_file:
            txt=txt_file.read()
        with open(os.path.join(current_folder,f'{file_name}.txt'),'w') as outfile:
            outfile.write(txt)
    
    print(f'Total file saved: {i+1}.')
    print('Data cleaned.')
        
def plot_duplicates(pairs,sub_size=3):
    n_pairs=len(pairs)
    fig,axes=plt.subplots(n_pairs,2,figsize=(2*sub_size,n_pairs*sub_size))
    for i in range(n_pairs):
        img1=Image.open(pairs[i][0])
        img2=Image.open(pairs[i][1])
        axes[i,0].imshow(img1)
        axes[i,1].imshow(img2)
        for j in range(2):
            axes[i,j].axis('off')

    plt.tight_layout()
    plt.savefig('../figures/duplicates.pdf')


if __name__ == "__main__":
    dir_name=laka_config.dir_name
    deduped_paths=get_deduped_path(dir_name)
    create_clean_data(deduped_paths)
