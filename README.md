# Model description
**MHC-I-EpiPred** (MHC-I-EpiPred, T cell MHC I molecular epitope prediction) is a protein language model fine-tuned from [**ESM2**](https://github.com/facebookresearch/esm) pretrained model [(***facebook/esm2_t33_650M_UR50D***)](https://huggingface.co/facebook/esm2_t33_650M_UR50D).    

**MHC-I-EpiPred** is a classification model that uses potential epitope peptides as input to predict T cell epitopes of MHC-I. The model is fed with a peptide sequence, and the output of the model is whether the peptide is a T cell epitope of MHC-I.  
  
# Dataset
The original data was downloaded from IEDB data base at https://www.iedb.org/home_v3.php.  
The full data can be downloaded at  https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip  
This dataset comprises 543,717 T-cell epitope entries, spanning a variety of species and infections caused by diverse viruses. The epitope information included encompasses a broad range of potential sources, including data relevant to disease immunotherapy.  

Finally, the dataset we used to train the model contains 41,060 positive and negative samples, which is stored in https://github.com/pengsihua2023/MHC-I-EpiPred/tree/main/data.   

# Results
**MHC-I-EpiPred** achieved the following results:  
Training Loss (mse): 0.1044  
Training Accuracy: 98.99%  
Evaluation Loss (mse): 0.1576  
Evaluation Accuracy: 97.04%  
Epochs: 492 

# Model at Hugging Face
https://huggingface.co/sihuapeng/MHC-I-TCEpiPred   

# How to use **MHC-I-EpiPred**
### An example
Pytorch and transformers libraries should be installed in your system.  
### Install pytorch
```
pip install torch torchvision torchaudio

```
### Install transformers
```
pip install transformers

```
### Run the following code
```
Coming soon!

```

## Funding
This project was funded by the CDC to Justin Bahl (BAA 75D301-21-R-71738).  
### Model architecture, coding and implementation
Sihua Peng  
## Group, Department and Institution  
### Lab: [Justin Bahl](https://bahl-lab.github.io/)  
### Department: [College of Veterinary Medicine Department of Infectious Diseases](https://vet.uga.edu/education/academic-departments/infectious-diseases/)  
### Institution: [The University of Georgia](https://www.uga.edu/)  

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64c56e2d2d07296c7e35994f/2rlokZM1FBTxibqrM8ERs.png)
