# CVND---Image-Captioning-Project

# Instructions  
1. clone this repo: https://github.com/cocodataset/cocoapi  
git clone https://github.com/cocodataset/cocoapi.git  

2. setup (also described in the readme here)  
cd cocoapi/PythonAPI  
make  
cd ..  

3. download some data from here: http://cocodataset.org/#download (described below)

* under **Annotations**, download:
  * 2014 Train/Val annotations [241MB] (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * 2014 Testing Image info [1MB] (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* under **Images**, download:
  * 2014 Train images [83K/13GB] (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * 2014 Val images [41K/6GB] (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * 2014 Test images [41K/6GB] (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb).
