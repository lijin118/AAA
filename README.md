
# Divergence-agnostic Unsupervised Domain Adaptation

This is the PyTorch code for validating our proposed method in this paper. For simple validation purpose, only a trained model and the validation code are provided for review, the full code will be published later. Here we take the task A->D on Office-31 dataset as an example. 

The framework of the proposed method is shown in Fig.2 of the manuscript.

## Prerequisites: 

* python == 3.6.10 
* pytorch ==  1.5.0 
* cuda == 9.2
* torchvision = 0.6.0


## The source code files and folders:

  * "network.py": contains the network architectures we used in this paper (including the backbone, the bottleneck, the classifier and the perturbation generator).
  * "data": contains the list files and the images of the testing data.
  * "data_list.py" re-implements the ImageList class in pytorch for loading the data with corresponding index. 
  * "loss.py": contains some loss functions we used in this paper.
  * "AD": contains the trained models (F, B and C) of the task A->D for validation.
  
  
# How to run the codes and validate the results of this paper

Limited by volume, we do not provide the Office-31 dataset. To validate the model, you may get the dataset [here](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view) and put it to the path according to "dslr_list.txt". After that, just run the main file: 

```
python main.py
```

Wait a moment and the results will be displayed in the console. 


