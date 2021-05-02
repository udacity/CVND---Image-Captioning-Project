#!/bin/bash

mkdir datak
cd datak
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/test2014.zip
wget http://images.cocodataset.org/annotations/image_info_test2014.zip
unzip annotations_trainval2014.zip
unzip train2014.zip
unzip val2014.zip
unzip image_info_test2014.zip
rm *.zip
