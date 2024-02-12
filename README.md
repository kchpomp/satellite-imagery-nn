# Satellite Imagery Multi-class Classification using Neural Networks

## Table of contents:
- ### [1. Description.](#item-one)
- ### [2. Used Libraries.](#item-two)
- ### [3. Evaluation Metrics.](#item-three)
- ### [4. Dataset creation.](#item-four)
- ### [5. Neural Networks models used.](#item-five)
- ### [6. Current Results.](#item-six)
- ### [7. Future Updates.](#item-seven)

<a id="item-one"></a>
## 1. Description.
> This project is dedicated to using the Neural Network Models to perform the Multi-class classification if Nitrogen Deficiency in the soil.
> 
> In the concept of smart farming and precision farming this project is aimed at optimization of data analysis process, fertilizers usage and as a consequence to yield increase.
>
> ### Workflow:
> - Review of contemporary techniques of smart farming;
> - Review of contemporary analysis methods of Satellite and Aero images;
> - Review of machine learning algorithms that are used for multi-band satellite and aero images;
> - Review of existing solutions of smart farming concept on the market;
> - Development of the Satellite Imagery Pipeline;
> - Development and Adaptation of contemporary neural network methods for detection of nitrogen deficiency in the soil;
> - Build of the custom dataset from the existing multi-band photos with application of augmentation techniques;
> - Training of the Neural Network models on the created custom dataset, estimation of the results, fine-tuning of hyper-parameters;
> - Implementation of K-Fold validation technique to obtain final result;
> - Conducting a comparative analysis of training results and selection of the best model.   
---
<a id="item-two"></a>
## 2. Used Libraries.
> - **PyTorch** for building Neural Network models;
> - **TIFFFILE** Python Library to process images in *.tiff format;
> - **Albumentations** Python Library for image augmentation;
> - **NumPY** and **OpenCV** for image processing;
---
<a id="item-three"></a>
## 3. Evaluation Metrics.
> **Accuracy** represents accuracy and is calculated as the ratio of correct predictions to the total number of predictions.<br>
> **Precision** is a metric that performs well in the presence of unbalanced classes. This metric uses concepts of binary classification and is calculated as the ratio of true positives (TP) to the sum of TP and false positives (FP).<br>
> **Recall** shows the proportion of samples from a class that were correctly predicted by the model, calculated as the ratio of TP to the sum of TP and correct negatives (FN).<br>
> 
> ### Calculated Metrics:
> - **F-1 score** is a combination of precision and recall and is calculated using formula (1) below:<br>
> $$F_1=\frac{2 \times Precision \times Recall}{(Precision+Recall)}$$ 
> - **IoU** or overlap coefficient is one of the most popular metrics used in detecting objects in an image. This metric is calculated as the **Jaccard** coefficient using formula (2), presented below:<br>
> $$IoU=\frac{TP}{TP+FN+FP}$$
> - Also, the **Jaccard error matrix** is often used to show how many objects were assigned to a certain class. Matrix elements located along the diagonal show correct predictions, while elements located along the diagonal show the number of elements incorrectly assigned to a particular class.
---
<a id="item-four"></a>
## 4. Dataset Creation.
> For the practical part, images of agricultural fields with spring wheat sown were used, which were taken with a **Geoscan-401** quadcopter. All images are combined into a high-resolution multi-channel image in **GeoTIFF** format.<br>
> ### Each image contains 7 channels:
> - **Red** Channel;
> - **Green** Channel;
> - **Blue** Channel;
> - Red Edge Region of Infrared Region **RedEdge**;
> - Near Infrared Region **NearIR**;
> - Normalized relative vegetation index **NDVI**;
> - Test sites marked by expert agronomists divided on 6 classes with Nitrogen level from 0 to 200 kg.
---
<a id="item-five"></a>
## 5. Neural Network models used.
> To solve the problem were used several modern Neural Network Architectures: U-Net and it's state-of-the-art modifications (Attention U-Net, U-Net++, U-Net3+), DeepLabV3+, FastFCN.
> ### Neural Network Architectures:
> - #### U-Net:
>> U-Net consists of two parts: encoder and decoder that are connected by skip connections, below is presented convolution block of Encoder;<br>
> 
>> ![](/U-Net%20Convolution%20Block.png "U-Net Encoder Convolution Block")
> - #### Attention U-Net:
>> Attention U-Net unlike original architecture has got instead of skip connections "Attention Mechanism" that allows to highlight only relevant regions of activated neurons in the process of learning;<br>
>> а) Attention Mechanism scheme б) Detailed representation of a convolutional block with a convolutional layer with a 3x3 kernel (convolutional block X) в) Detailed representation of a convolutional block with a convolutional layer with a 1x1 kernel (convolutional block Y).<br>
> 
>> ![](/Attention%20Mechanism.png "Attention Mechanism for Attention U-Net")
> - #### U-Net++:
>> U-Net++ main difference from the original U-Net is usage weighted convolutional blocks instead of skip connections between the encoder and decoder;<br>
>> а) Architecture of the U-Net++ Neural Network model б) Detailed view of the convolutional block.<br>
>
>> ![](/U-Net++.png "U-Net++ Architecture")
> - #### U-Net3+:
>> U-Net3+ has got several differences from the original U-Net architecture: U-Net3+ saves information from all skip connections; Thus, the decoder layer, in addition to the signal from the lower encoder layer, also receives all skip connections from all encoder layers. This allows you to save more general information about the object; Each connection consists of a convolutional block, and also, if there is a transition to a higher level, then bilinear interpolation is applied, and if there is a transition to a lower level, then the subsampling operation (max pooling) is applied;<br>
>> а) Architecture of the U-Net3++ Neural Network model б) Detailed view of a convolutional block.<br>
>
>> ![](/U-Net3+.png "U-Net3+ Architecture")
> - #### DeepLabV3+:
>>  DeepLabV3+ also consists of the Encoder and Decoder. Encoder consists of the two blocks: backbone that consists of the ResNet Architecture type neural network block and ASPP (Atrous Spatial Pyramid Pooling) block. Below is presented architecture of the ASPP block and common DeepLabV3+ architecture.<br>
>
>> ![](/ASPP.png "Atrous Spatial Pyramid Pooling block")
>> ![](/DeepLabV3Plus.png "DeepLabV3+ Architecture")
> #### FastFCN:
>> Like all aforementioned models FastFCN also has got explicit Encoder-Decoder architecture. Moreover, the architecture itself is similar to the DeepLabV3+, however, instead of the ASPP block is used JPU (Joint Pyramid Upsampling) block. However, it has more "linear" model itself.
>> ![](/JPU.png "Joint Pyramid Upsampling block")
>> ![](/FastFCN.png "FastFCN Architecture")
---
<a id="item-six"></a>
## 6. Current Results.
> |Neural Network Model| U-Net | Attention U-Net | U-Net++ | U-Net3+ | DeepLabV3+ | FastFCN |
> |:-------------------|:-----:|:---------------:|:-------:|:-------:|:----------:|:-------:|
> | Mean Accuracy      |0.9354 |     0.9554      |  0.9818 |  0.9617 |   0.9773   |  0.9758 |
> | Mean IoU           |0.8790 |     0.9151      |  0.9644 |  0.9262 |   0.9480   |  0.9527 |
> | Mean Precision     |0.9366 |     0.9583      |  0.9817 |  0.9633 |   0.9750   |  0.9767 |
> | Mean Recall        |0.9367 |     0.9550      |  0.9850 |  0.9617 |   0.9750   |  0.9750 |
> | Mean F-1 Score     |0.9350 |     0.9567      |  0.9817 |  0.9600 |   0.9717   |  0.9783 |
> | Training time, h   | 40.3  |       47.4      |   67.3  |   93.2  |    21      |   57    |
>
>> For training was used RTX 3060 GPU with 12 Gb of memory. 
>> According to the results the best model for solution is U-Net++. 
---
<a id="item-seven"></a>
## 7. Future Updates.
> ### There is a list of the possible updates:
> - Usage of the random generated pixels to enhance the dataset;
> - New Neural Networks Architectures;
> - Modification of ASPP and JPU blocks;
> - Creation of the custom Neural Network Architecture;
> - Embedding the resulting solution in the automated satellite image classification system;
> - Add different sites;
> - Add different agricultures;
---