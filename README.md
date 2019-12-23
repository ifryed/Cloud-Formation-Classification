
# DeepLearning Project: Cloud Formation Detection  
  
# Data  
The data was gathered from this Kaggle compatition [ Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization/data)  
## Prerequisites   
* Python 3.6  
* Opencv 4.x  
* TensorFlow V1.15  

## Preparing data
Download the data from [here](https://www.kaggle.com/c/understanding_cloud_organization/data). Extract them to the 'data' folder, and run from the folder:

    python data_gen.py

  
## Usage:  
	usage: main.py [-h] --model MODEL [--batch_size MINI_BATCH]  
	 [--samples SAMPLES] [--use_gpu GPU] [--gpu_full FULL_GPU] [--weights WEIGHTS_PATH] 

    optional arguments:  
      -h, --help            show this help message and exit  
      --model MODEL         Which model to use? (SLP,ANN,CNN)  
      --batch_size MINI_BATCH  
                            Mini Batch size  
      --samples SAMPLES     How many samples to load from each catagory  
      --use_gpu GPU         Use GPU?  
      --gpu_full FULL_GPU   Test on full test when using GPU?  
      --weights WEIGHTS_PATH  
                            Location of weights  

## Authors  
[Naomi Tal Tsabari](https://github.com/naomital)  
[Shai Aharon](https://github.com/ifryed)
