import os
import requests
from bs4 import BeautifulSoup
import base64
import re
import urllib.parse
import json
import itertools

from tqdm import tqdm

import hashlib
import imagehash

from PIL import Image
import torch
from torchvision import transforms

import laka_config

def extract_caption_and_links(page_url, base_url,session):
    """Extract captions (clickable text) and their links from the current result page."""
    response = session.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    captions_and_links = []
    
    for li_tag in soup.find_all("li"):
        caption = li_tag.get_text(strip=True)
        a_tag = li_tag.find("a",href=True)
        link = urllib.parse.urljoin(base_url, a_tag['href'])
        captions_and_links.append((caption, link))
    
    return captions_and_links

def extract_blob_image(item_url, session):
    """Extract the blob-format image from an item's page and convert it to a regular image."""
    response = session.get(item_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Attempt to find the image in a <script> or <canvas> tag
    blob_data = None
    
    # Example: Look for base64-encoded image data (e.g., <img src="data:image/png;base64,...">)
    img_tag = soup.find("img", {"src": re.compile(r"^data:image/")})
    if img_tag:
        blob_data = img_tag['src']
    else:
        # Search for the blob in <script> tags or other elements
        script_tag = soup.find("script", string=re.compile(r"data:image/"))
        if script_tag:
            match = re.search(r"data:image/[^\"']+", script_tag.string)
            if match:
                blob_data = match.group(0)
    
    if blob_data:
        # Decode base64 image data
        header, encoded = blob_data.split(",", 1)  # Split the data URL
        file_extension = header.split(";")[0].split("/")[1]  # Extract file extension
        image_data = base64.b64decode(encoded)
        return image_data, file_extension
    
    return None, None

def get_file_name(caption):
    file_name=caption.split(';')[0]
    for symbol in ['!','@','#','%','^','&','*','(',')','~',';',':','/',r'\\','.',',','?','-','+','=',"'",' ',"'"]:
        if symbol==' ':
            file_name=file_name.replace(' ','_')
        else:
            file_name=file_name.replace(symbol,'')
    return file_name

def download_image(image_data, file_extension, caption,dir_name):
    """Save the decoded image and its caption."""
    # Create unique file name
    file_name=get_file_name(caption)[:50]
    image_name = f"{file_name}.{file_extension}"  # Limit caption length
    image_path = os.path.join(dir_name, image_name)
    
    # Save the image
    with open(image_path, "wb") as img_file:
        img_file.write(image_data)
    
    # Save the caption
    caption_path = os.path.join(dir_name, f"{file_name}.txt")
    with open(caption_path, "w") as caption_file:
        caption_file.write(caption)
    

def get_pagination_links(page_url, base_url,session):
    """Extract pagination links from the current page."""
    response = session.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    pagination_links = []
    
    for a_tag in soup.find_all("a", href=True):
        if "page=" in a_tag['href']:  # Assuming 'page=' identifies pagination links
            full_url = urllib.parse.urljoin(base_url, a_tag['href'])
            pagination_links.append(full_url)
    page_links= list(set(pagination_links)) 
    return page_links # Remove duplicates

def get_all_images(session,base_url,dir_name,params):
    location=params['location']
    subject=params['keyword2']
    year=params['term']
    print(f"Started for {location}/{subject}/{year}")

    # Queue for pages to process
    initial_url = f"{base_url}?location={location}&keyword2={subject}&term={year}&combined=on&suchwort=#selectie"
    pages_to_process = [initial_url]
    processed_pages = set()
    file_names=set()
    dir_name=os.path.join(dir_name,f'{location}_{subject}_{year}')
    os.makedirs(dir_name,exist_ok=True)
    
    while pages_to_process:
        # Process the next page
        current_page = pages_to_process.pop(0)
        if current_page in processed_pages:
            continue
        processed_pages.add(current_page)
        #print(f"Processing page: {current_page}")
        
        # Step 1: Extract captions and links
        captions_and_links = extract_caption_and_links(current_page,base_url, session)
        
        # Step 2: Visit each item page and process images
        for caption, item_url in captions_and_links:
            file_name=caption.split(';')[0].replace('/','_').replace('!','_').replace(' ','_').replace('.','_').replace('#','_')
            if file_name not in file_names:
                file_names.add(file_name)
                #print(f"Processing item: {caption}")
                image_data, file_extension = extract_blob_image(item_url, session)
                if image_data:
                    download_image(image_data, file_extension, caption,dir_name)
                else:
                    print(f"No image found for item: {caption}")
        
        # Step 3: Find and queue pagination links
        new_pages = get_pagination_links(current_page, base_url,session)
        for page in new_pages:
            if page not in processed_pages:
                pages_to_process.append(page)
        
        pages_to_process=list(set(pages_to_process))
        print(f"Length of the pending links: {len(pages_to_process)}")
    
    print(f"Finished for {location}/{subject}/{year}")
    print(f"Number of images: {len(file_names)}")
    
    return len(file_names)

def get_dropdown_options(session, dropdown_url):
    """Fetch options for dropdowns."""
    response = session.get(dropdown_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Identify the dropdowns and extract options
    dropdown_data = {}
    for select_tag in soup.find_all("select"):
        name = select_tag.get("name")
        options = [option.get("value") for option in select_tag.find_all("option") if option.get("value")]
        dropdown_data[name] = options
    
    return dropdown_data

def main(base_url,dir_name,progess_file):
    os.makedirs(dir_name, exist_ok=True)
    keyword_file_path=os.path.join(dir_name,progess_file)
    with requests.Session() as session:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
            })
        dropdown_data = get_dropdown_options(session, base_url)

        # Generate all combinations of dropdown values
        param_names = dropdown_data.keys()

        cnts = list(map(len,map(set,dropdown_data.values())))
        
        total=1
        for cnt in cnts:
            total*=cnt
        print(f"TOTAL COMBINATIONS: {total}")


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

def find_duplicates(valid_paths,dtype='text'):
    contents=[]
    duplicates=[]
    
    if dtype=='text':
        for txt_path in valid_paths:
            with open(txt_path,'r') as txt_file:
                txt=txt_file.read()
                if txt not in contents:
                    contents.append(txt)
                else:
                    duplicates.append(txt_path)
    elif dtype=='image':
        transform=transforms.ToTensor()
        for img_path in valid_paths:
            with Image.open(img_path) as img:
                img=transform(img).flatten()
                if img not in contents:
                    contents.append(img)
                else:
                    duplicates.append(img_path)
    else:
        raise Exception('No such data type.')

    print(f'Number of duplicates for {dtype}: {len(duplicates)}')
    
    return duplicates

def deduplicates(path):
    valid_img_paths,valid_txt_paths=get_valid_paths(path)
    print(f'Total number of images: {len(valid_img_paths)}')
    
    txt_duplicates=find_duplicates(valid_txt_paths,dtype='text')
    img_duplicates=find_duplicates(valid_img_paths,dtype='image')

    assert len(txt_duplicates)==len(img_duplicates),"Should be the same length."

    txt_duplicates=list(map(lambda x: os.path.splitext(x)[0],txt_duplicates))
    img_duplicates=list(map(lambda x: os.path.splitext(x)[0],img_duplicates))
    txt_duplicates.sort()
    img_duplicates.sort()

    if txt_duplicates==img_duplicates:
        print('Correct duplicates')

    else:
        print("Incorrect duplicates")



if __name__ == "__main__":
    base_url=laka_config.base_url
    dir_name=laka_config.dir_name
    progess_file=laka_config.progress_file
    main(base_url,dir_name,progess_file)
    #deduplicates(dir_name)
