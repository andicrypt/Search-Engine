# Landmark Image Search Engine

There are a database D and train dataset T that both contain VN landmark images. Create a image retrieval model such that given a image query, find the related images corresponding to that image query.

**Our method**
* Initially, we fine-tune *Vision Transformer* pre-trained model with our train dataset in order to create the VN Landmark classifier.
* Secondly, we classify all images of database by our trained classifier.
* For each image query, we classify that image and then within that class, we use *Local Feature Matching* to find the most related images and its precision score.



