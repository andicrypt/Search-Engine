import cv2
import numpy as np
import pickle 
from sklearn.metrics.pairwise import cosine_similarity

def local_feature(img_path):
    print('halolo', img_path)
    sift = cv2.SIFT_create()
    img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
    kp, des = sift.detectAndCompute(img, None)
    return (kp, des)

def local_feature_matching_score(des1, des2):
    cos_sim = cosine_similarity(des1, des2)
    return np.mean(cos_sim)

def load_descriptor(pickle_file):
    with open(pickle_file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

