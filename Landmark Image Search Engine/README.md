# Landmark Image Search Engine

There are a database D and train dataset T that both contain VN landmark images. Create a image retrieval model such that given a image query, find the related images corresponding to that image query.

**Our method**
* Initially, we fine-tune *Vision Transformer* pre-trained model with our train dataset in order to create the VN Landmark classifier.
* Secondly, we classify all images of database by our trained classifier.
* For each image query, we classify that image and then within that class, we use *Local Feature Matching* to find the most related images and its precision score.

**Contribution**
	- Vision Transformer classifier: I build this classifier separately in colab (I attach the ipynd in the submission), the trained model is created and used for later process.
	- Preclassify images of the database by our trained classifier.
	- Extract local features of images of database and query by SIFT.
	- GUI.

**Dataset** for fine-tuning *Vision Transformer* is provided in the following link: https://drive.google.com/file/d/1nQIfFMeQuq2rmYHcYAZGMqpROP4s8kAO/view?usp=sharing


**Video Demo**: I record a demo video and post it in Youtube: https://www.youtube.com/watch?v=Y1j7aNq1ygI  
	
