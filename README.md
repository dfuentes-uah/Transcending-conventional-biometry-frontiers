# Transcending-conventional-biometry-frontiers
In the first half of the 20th century, a first pulse oximeter was available to measure blood flow changes in the peripheral vascular net. However, it was not until recent times the PhotoPlethysmoGraphic (PPG) signal used to monitor many physiological parameters in clinical environments. Over the last decade, its use has extended to the area of biometrics, with different methods that allow the extraction of characteristic features of each individual from the PPG signal morphology, highly varying with time and the physical states of the subject. In this paper, we present a novel PPG-based biometric authentication system based on convolutional neural networks. Contrary to previous approaches, our method extracts the PPG signal's biometric characteristics from its diffusive dynamics, characterized by geometric patterns image in the (p, q)-planes specific to the 0-1 test. The diffusive dynamics of the PPG signal are strongly dependent on the vascular bed's biostructure, which is unique to each individual, and highly stable over time and other psychosomatic conditions. Besides its robustness, our biometric method is anti-spoofing, given the convoluted nature of the blood network. Our biometric authentication system reaches very low Equal Error Rates (ERRs) with a single attempt, making it possible, by the very nature of the envisaged solution, to implement it in miniature components easily integrated into wearable biometric systems. 
![Alt text](diagrama_biometria.png?raw=true "System Blocks Diagram")

# Working Conditions
This code runs on Tensorflow 2.0
# Train
To train and test the system it is necessary to download the associated database and pretrained models from the following URL:

https://kaggle.com/datasets/c0e959f62d846c46f0001ece57aad55b519da03d0431a0ab5cc5d12fe6eb92a2

The Data have to be stored in data folder and divided by the image modality(crudas, filtradas, normalizadas). In the case of the pretrained models are stored in the models folder.

Once you adapted the data to the folders system, you'll only have to run main.py adjusting the internal params of the script to train or test a model and it will train or provide an scoring of the model.

