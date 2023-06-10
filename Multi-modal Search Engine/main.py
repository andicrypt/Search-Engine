import streamlit as st 
from PIL import Image
from encoder import encoding_text, encoding_image
import faiss
import pickle
import numpy as np
from tqdm import tqdm
import random
import os

title = 'Multi-modal search'
st.set_page_config(page_title=title, layout="wide")
st.title(title)

dataset_dir = r'C:\Users\THINKPAD\Downloads\val2017\val2017'
image_filenames = [filename for filename in os.listdir(dataset_dir)]
image_filenames = random.sample(image_filenames, k=200)

text_search = st.text_input(label='Input text query')


@st.cache
def faiss_indexer():
    indexer = faiss.IndexFlatL2(512)
    for i in tqdm(range(len(image_filenames))):
        encoded_image = encoding_image(os.path.join(dataset_dir, image_filenames[i]))
        indexer.add(encoded_image)
    return indexer
indexer = faiss_indexer()
    

def display_images(filenames):
    container = st.container()
    num_of_cols = 8
    cols = container.columns(num_of_cols)
    max_images = 32
    for i, filename in enumerate(filenames):
        if i > max_images:
            break
        with cols[i % num_of_cols]:
            full_path = os.path.join(dataset_dir, filename)
            img = Image.open(full_path)
            st.image(img, use_column_width=True)
        
@st.cache
def image_text_search(text_search):
    ## Encoding search text
    encoded_text = encoding_text(text_search)
    
    ## Searching
    f_dists, f_ids = indexer.search(np.array(encoded_text), k=20)
    result_ids = f_ids[0][1:]
    result_files = [image_filenames[i] for i in result_ids]
    return result_files


if text_search is not None:
    result_files = image_text_search(text_search)
    display_images(result_files)
else: 
    display_images(image_filenames)
    