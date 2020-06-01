# CVND---Image-Captioning-Project

# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below) or download using torrent from http://academictorrents.com/details/f993c01f3c268b5d57219a38f8ec73ee7524421a

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).

# Other Download Methods
Use these links to download files faster 
Torrent download (**Images ONLY**): http://academictorrents.com/details/f993c01f3c268b5d57219a38f8ec73ee7524421a

Google Drive (**Annotations ONLY**): https://bit.ly/coco-annotations
Google Drive (**Test Images**): https://bit.ly/coco-test2014
Google Drive (**Train Images**): https://bit.ly/coco-train2014
Google Drive (**Validation Images**): https://bit.ly/coco-val2014

OWNER: http://cocodataset.org/

Terms: By Downloading from links provided you agree to these terms: http://mscoco.org/terms_of_use/

