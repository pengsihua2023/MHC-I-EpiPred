# MHC-I-EpiPred-ESM2
## Model descriptions
**PPPSL-ESM2**(PPPSL, Prediction of prokaryotic protein subcellular localization) is a protein language model fine-tuned from [**ESM2**](https://github.com/facebookresearch/esm) pretrained model [(***facebook/esm2_t36_3B_UR50D***)](https://huggingface.co/facebook/esm2_t36_3B_UR50D) on a prokaryotic protein subcellular localization dataset.   

**PPPSL-ESM2** achieved the following results:  
Train Loss: 0.0015  
Train Accuracy: 0.9893  
Validation Loss: 0.0184  
Validation Accuracy: 0.9702  
Epoch: 20 
# The dataset for training **PPPSL-ESM2**
The full dataset contains 11,970 protein sequences, including Cellwall (87), Cytoplasmic (6,905), CYtoplasmic Membrane (2,567), Extracellular (1,085), Outer Membrane (758), and Periplasmic (568).
The highly imbalanced sample sizes across the six categories in this dataset pose a significant challenge for classification.  

The dataset was downloaded from the website at [**DeepLocPro - 1.0**](https://services.healthtech.dtu.dk/services/DeepLocPro-1.0/). 

# Model training code at GitHub
https://github.com/pengsihua2023/PPPSL-ESM2

# How to use **PPPSL-ESM2**
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "sihuapeng/PPPSL-ESM2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample protein sequence
protein_sequence = "MSKKVLITGGAGYIGSVLTPILLEKGYEVCVIDNLMFDQISLLSCFHNKNFTFINGDAMDENLIRQEVAKADIIIPLAALVGAPLCKRNPKLAKMINYEAVKMISDFASPSQIFIYPNTNSGYGIGEKDAMCTEESPLRPISEYGIDKVHAEQYLLDKGNCVTFRLATVFGISPRMRLDLLVNDFTYRAYRDKFIVLFEEHFRRNYIHVRDVVKGFIHGIENYDKMKGQAYNMGLSSANLTKRQLAETIKKYIPDFYIHSANIGEDPDKRDYLVSNTKLEATGWKPDNTLEDGIKELLRAFKMMKVNRFANFN"

# Encode the sequence as model input
inputs = tokenizer(protein_sequence, return_tensors="pt")

# Perform inference using the model
with torch.no_grad():
    outputs = model(**inputs)

# Get the prediction result
logits = outputs.logits
predicted_class_id = torch.argmax(logits, dim=-1).item()
id2label = {0: 'CYtoplasmicMembrane', 1: 'Cellwall', 2: 'Cytoplasmic', 3: 'Extracellular', 4: 'OuterMembrane', 5: 'Periplasmic'}
predicted_label = id2label[predicted_class_id]

# Output the predicted class
print ("===========================================================================================================================================")
print(f"Predicted class Label: {predicted_label}")
print ("===========================================================================================================================================")

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
