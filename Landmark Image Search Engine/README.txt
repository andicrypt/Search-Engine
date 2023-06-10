19125084	- 	Tran Hai Anh Dien

Topic: Vietnam Landmark Retrieval by image query 

### My model: Consist of two things:
	- Vision Transformer classifier (trained by train dataset), used for pre-classify images of the database
	- Extract local features of images by SIFT and use cosine measure to score the similarity
	- Build Streamlit GUI for my app.

### Contribution:
	- Vision Transformer classifier: I build this classifier separately in colab (I attach the ipynd in the submission), the trained model is created and used for later process.
	- Preclassify images of the database by our trained classifier.
	- Extract local features of images of database and query by SIFT.
	- GUI.


### Submission: includes the following files
	- Fine_tune_ViT_Classifier.ipynb (used for fine-tuning ViT classifier with our train dataset)
	- main.py (main file containing streamlit code)
	- local_features.py (additional functions for local features and scoring)
	- config.py (config variables)
	- dataframe.pkl (pre-classify database images results)
	- requirements.txt (modules involved)

	Folder:
	- local_feature: containing (1200 pickle file containing local feature of 1200 database images) (I push this folder in Github with link: https://github.com/andicrypt/Search-Engine/tree/main/Landmark%20Image%20Search%20Engine)


### VIDEO: I record a demo video and post it in Youtube: https://www.youtube.com/watch?v=Y1j7aNq1ygI  
	Because this youtube video has low quality; therefore I also post this video in Drive: https://drive.google.com/file/d/17dNHPLVQpLvCufL-Uj7EkBmbPNUfq290/view?usp=sharing




	
	