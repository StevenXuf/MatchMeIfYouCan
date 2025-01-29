import os
import requests
import base64
import re
import urllib.parse
import json
import itertools
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from tqdm import tqdm

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
    #subject=params['keyword2']
    #year=params['term']
    print(f"Started for {location}")

    # Queue for pages to process
    initial_url = f"{base_url}?location={location}&suchwort=#selectie"
    pages_to_process = [initial_url]
    processed_pages = set()
    file_names=set()
    dir_name=os.path.join(dir_name,f'{location}')
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
    
    print(f"Finished for {location}")
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
    if os.path.exists(keyword_file_path):
        print('File exists')
        with open(keyword_file_path,'r') as f:
            finished_combinations=json.load(f)
    else:
        print('File does not exist')
        finished_combinations=[]

    # Start a session
    with requests.Session() as session:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
            })
        dropdown_data = get_dropdown_options(session, base_url)
        
        # Generate all combinations of dropdown values
        param_names = list(dropdown_data.keys())[0]
        countries=list(map(set,dropdown_data.values()))[0]
        
        total=0
        for country in tqdm(countries):
            if country not in finished_combinations:
                params = dict(location=country)
                n_imgs=get_all_images(session,base_url,dir_name,params)
                total+=n_imgs
                finished_combinations.append(country)
                with open(keyword_file_path,'w') as f:
                    json.dump(finished_combinations,f)
        print(f"SCRAPING FINISHED \nTOTAL NUMBER OF IMAGES: {total}")


if __name__ == "__main__":
    base_url=laka_config.base_url
    dir_name=laka_config.dir_name
    progess_file=laka_config.progress_file
    main(base_url,dir_name,progess_file)
