# CRACK DETECTION AND SEGMENTATION FOR INFRATRUCTURAL  ANALYSIS

We present our solution to the problem crack segmentation for pavements and walls. This project has been completed as a part of our End-Semester Evaluation for the Course Deep Learning for Signal and Image Processing.

Our approach is based on the UNet network with transfer learning on the two popular architectures: VGG16 and Resnet.
The result shows that a large crack segmentation dataset helps improve the performance of the model in diverse cases that 
could happen in practice.

# Contents
 - [Overview](#Overview)
 - [Dataset](#Dataset)
 - [Dependencies](#Dependencies)
 - [Test Images Collection](#Test-Images-Collection)
 - [Inference](#Inference)
 - [Training](#Training)
 - [Result](#Result)
 - [Citation](#Citation)



# Overview
Our project focuses on the segmentation of cracks in infrastructure using deep learning techniques, aiming to enhance the efficiency and accuracy of crack detection and analysis.​
This aims to provide a tool for infrastructure maintenance, facilitating timely repairs and preventing potential hazards, thus ensuring the longevity and safety of civil structures without much expenditure

# Dataset
It contains around 11,200 images that are merged from 12 available crack segmentation datasets.​

The name prefix of each image is assigned to the corresponding dataset name that the image belong to. ​

All the images are resized to the size of (448, 448).​

The two folders called images and masks contain all the images. Two folders called train and test contain training and testing images split from the above folder. 

***
# Dependencies
```python
conda create --name crack
conda install -c anaconda pytorch-gpu 
conda install -c conda-forge opencv 
conda install matplotlib scipy numpy tqdm pillow
```

***
# Inference
- take the pre-trained model
- put the model under the folder ./models
- run the code
```pythonstub
python inference_unet.py  -in_dir ./test_images -model_path ./models/model_unet_resnet_101_best.pt -out_dir ./test_result
```

***
# Test Images Collection
The model works quite well in situations where there are just almost crack pixels and the concrete background in the images. 
However, it's often not the case in reality, where lots of different objects could simultenously show up in an image. 
Therefore, to evaluate the robustness of the crack model, we tried to come up with several cases that could happen in practice. 
These images could be found in the folder ./test_imgs in the same repository 

- pure crack: these are ideal cases where only crack objects occur in the images.
- like crack: pictures of this type contains details that look like crack 
- crack with moss: there is moss on crack. These cases occur a lot in reality.
- crack with noise: the background (wall, concrete) are lumpy  
- crack in large context: the context is large and diverse. For example, the whole house or street with people



# Training
- step 1. download the dataset from [the link](https://drive.google.com/open?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP)
- step 2. run the training code
- step 3: 
```python 
python train_unet.py -data_dir PATH_TO_THE_DATASET_FOLDER -model_dir PATH_TO_MODEL_DIRECTORY -model_type resnet_101
```

# Result
The best result is achieved by UNet_VGG_16 

***

# Citation
Note: please cite the corresponding papers when using these datasets.

CRACK500:
>@inproceedings{zhang2016road,
  title={Road crack detection using deep convolutional neural network},
  author={Zhang, Lei and Yang, Fan and Zhang, Yimin Daniel and Zhu, Ying Julie},
  booktitle={Image Processing (ICIP), 2016 IEEE International Conference on},
  pages={3708--3712},
  year={2016},
  organization={IEEE}
}' .

>@article{yang2019feature,
  title={Feature Pyramid and Hierarchical Boosting Network for Pavement Crack Detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={arXiv preprint arXiv:1901.06340},
  year={2019}
}

GAPs384: 
>@inproceedings{eisenbach2017how,
  title={How to Get Pavement Distress Detection Ready for Deep Learning? A Systematic Approach.},
  author={Eisenbach, Markus and Stricker, Ronny and Seichter, Daniel and Amende, Karl and Debes, Klaus
          and Sesselmann, Maximilian and Ebersbach, Dirk and Stoeckert, Ulrike
          and Gross, Horst-Michael},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  pages={2039--2047},
  year={2017}
}

CFD: 
>@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}

AEL: 
>@article{amhaz2016automatic,
  title={Automatic Crack Detection on Two-Dimensional Pavement Images: An Algorithm Based on Minimal Path Selection.},
  author={Amhaz, Rabih and Chambon, Sylvie and Idier, J{\'e}r{\^o}me and Baltazart, Vincent}
}

cracktree200: 
>@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}

>https://github.com/alexdonchuk/cracks_segmentation_dataset
>https://github.com/khanhha/crack_segmentation



>https://github.com/yhlleo/DeepCrack

>https://github.com/ccny-ros-pkg/concreteIn_inpection_VGG


***

# Contributors

> [Amirthavarshini V](https://github.com/Amirtha2503)

