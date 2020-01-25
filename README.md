
# DeepLearning Project: Cloud Formation Classification  
  
# Data  
The data was gathered from this Kaggle compatition [ Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization/data)  
## Prerequisites   
- Python 3.6  
- TensorFlow V2.x (For Perceptron/SimpleAnn (used in main.py) TF V1.15)  
- Opencv 4.x  
- Pandas  
- matplotlib  
- dataclasses  
- tqdm  
  
## Contents  
- [Prepare Data](#preparedata) 
- [Single/Multi-layer NN](#singlemulti-layer-nn)  
- [CNN](#cnn)  
- [AutoEncoder/KNN](#autoencoderknn)  
- [Auxiliary Loss](#auxiliary-loss)

### Prepare Data
First dowload the data from [ Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization/data) and unpack it to the `data` folder.
Then run 

`python data/data_gen.py`

This will extract the data into folders by there class, where each image contains one cloud formation only.
  
### Single/Multi-layer NN
**For now, main.py works with TF 1.15**

To use run the SLP or MLP, run the `main.py` with the following arguments:

    usage: 
    python main.py [-h] --model MODEL [SLP,ANN,CNN] [--batch_size MINI_BATCH]
                   [--samples SAMPLES] [--use_gpu GPU] 
                   [--gpu_full FULL_GPU] [--weights WEIGHTS_PATH]

### CNN
To use run the CNN, run the `CNN.py`:

    usage:
    python CNN.py
    
### AutoEncoder/KNN
To use the AutoEndocer/KNN, run:

    python autoEncoder.py
    python classify_knn.py --model [PATH_TO_SAVED_MODEL]/encoder --images PATH_TO_MINI_DATA

### Auxiliary Loss
To use the final (best results) model with the AE and auxiliary loss run:

    python auxiliary_loss.py

  
## Authors  
[Naomi Tal Tsabari](https://github.com/naomital)  
[Shai Aharon](https://github.com/ifryed)
