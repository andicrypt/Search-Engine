from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os
import pickle
from config import labels
from local_features import local_feature_matching_score, local_feature, load_descriptor
import random

# start streamlit
st.set_page_config(page_title="Vietnam landmark image retrieval", layout="wide")
st.title("Vietnam landmark image retrieval")


database_dir = r"C:\Users\THINKPAD\Downloads\Landmark_Retrieval\Landmark_Retrieval\test\database"
db_img_paths = [os.path.join(database_dir, filename) for filename in os.listdir(database_dir) ]

query_dir = r"C:\Users\THINKPAD\Downloads\Landmark_Retrieval\Landmark_Retrieval\test\query"
qr_img_paths = [os.path.join(query_dir, filename) for filename in os.listdir(query_dir)]
qr_filenames = [filename for filename in os.listdir(query_dir)]



# to open a pickle file storing your list
@st.cache
def open_pickle(filename: str):
    dataframe = pd.read_pickle(filename)
    return dataframe
dataframe = open_pickle('dataframe.pkl')    

qfile = st.file_uploader(label='Input query image', accept_multiple_files=False)




def display_result(df):
    container = st.container()
    num_of_cols = 8
    cols = container.columns(num_of_cols)
    max_images = 64
    print_score = 'score' in df
    for i, row in enumerate(df.itertuples()):
        if i > max_images:
            break
        with cols[i % num_of_cols]:
            full_path = os.path.join(database_dir, row.image_name)
            img = Image.open(full_path)
            st.image(img, use_column_width=True)
            st.write(row.image_name)
            if print_score:
                st.write(f"score: {row.score}")

@st.cache
def setup(model_dir):
    # Load ViT classifier model
    model = ViTForImageClassification.from_pretrained(model_dir)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
    model.eval()
    return model, feature_extractor

model_dir = r'C:\Users\THINKPAD\Downloads\classifier\classifier'
model, feature_extractor = setup(model_dir)

@st.cache
def preprocessing(img_path, model, feature_extractor) -> int:
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    input = feature_extractor(images=img, return_tensors='pt')
    output = model(**input)

    # prediction
    logits = output.logits
    predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

@st.cache
def scoring_local_features(query_image, dataframe):
    query_kp, query_des = local_feature(os.path.join(query_dir, query_image))
    dataframe['score'] = None
    local_feature_dir = r'C:\Users\THINKPAD\Desktop\university\Semester 11\IR\HW5 VN Landmark Retrieval\streamlit\local_feature\content\local_feature'
    for i, row in dataframe.iterrows():
        file_path = os.path.join(local_feature_dir, row.image_name + '.pkl')
        db_des = load_descriptor(file_path)
        dataframe.loc[i, 'score'] = local_feature_matching_score(query_des, db_des)
    dataframe = dataframe.sort_values(by='score', ascending=False)
    return dataframe





qfile_name = None
if qfile is not None :
    qfile_name = qfile.name
        
    with st.spinner('Wait for it...'):
        idx = preprocessing(os.path.join(query_dir, qfile_name), model, feature_extractor)
    st.success('Done!')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open(os.path.join(query_dir, qfile_name)), caption=qfile_name)
    with col2:
        st.markdown(f'***Prediction: {labels[idx]}***')
        lfm = st.button('Local Feature Matching')
    
    st.write('Similar Images: ')
    result_df = dataframe[dataframe['prediction']==idx]
    if lfm == True:
        result_df = scoring_local_features(qfile_name, result_df)
    display_result(result_df)
else:
    display_result(dataframe)
    