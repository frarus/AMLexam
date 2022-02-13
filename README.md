# AML 2021/2022 "Real-time Domain Adaptation in Semantic Segmentation" Project
This folder contains the full code of the project for the final exam Advanced Machine Learning at Politecnico di Torino. 
<br>
This contains:
1. An implementation of a real-time semantic segmentation network, BiSeNet (Bilateral Segmentation Network), that can exploit two different backbones, ResNet-101 or ResNet-18;
2. An implementation of an unsupervised adversarial domain adaptation algorithm;
3. A variation of the unsupervised adversarial domain adaptation algorithm with lightweight depthwise-separable convolutions for the adversarial discriminator, which significantly reduce the total number of parameters and the total number of Floating Point Operations (FLOPS) of the model, making it suitable for mobile mobile and/or embedded devices;
4. Two image-to-image transformations, to improve domain adaptation:
    * FDA;
    * LAB;
5. Generation of pseudo-labels for the target domain, to further enhance domain adaptation. 
6. A combination of LAB transformation and pseudo labels generation for the target domain.
</br>
The network BiSeNet is based on pytorch 0.4.1 and python 3.6

The notebook "Real-time Domain Adaptation in Semantic Segmentation.ipynb" includes all the results, except for the last one that is the "Real-time Domain Adaptation in Semantic Segmentation.ipynb (2)" notebook

## Train
1. To train the the network BiSeNet:
```
python train.py
```  
2. To train the adversarial algorithm:
```
python trainGTAtoCityscapes.py
```  
3. To train using the lighweight discriminator:
```
python trainGTAtoCityscapes.py --use_DSC, 1
```  
4. To train applying the FDA transformation on source data:
```
python trainGTAtoCityscapes.py --use_DSC, 1 \
                               --transformation_on_source, FDA
```  
To train applying the LAB transformation on source data:
```
python trainGTAtoCityscapes.py --use_DSC, 1 \
                               --transformation_on_source, LAB
```  
5. To train applying the the FDA transformation on source data:
```
python trainGTAtoCityscapesSSL.py 
                               
```  
5. To train applying the the FDA transformation on source data:
```
python trainGTAtoCityscapesSSL.py --transformation_on_source, LAB
                               
```  


## Test
To evaluate the BiSeNet model:
```
python eval.py
```
