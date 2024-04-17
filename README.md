# ArchAIDE: an AI tool for Automatic Recognition of Archaeological Ceramic Sherds 

ArchAIDE is an EU project, led by University of Pisa, focused on the development of methodologies for the automatic recognition of archaeological pottery. 
It utilizes artificial intelligence and machine learning techniques to analyze images of pottery sherds and identify their typology, helping archaeologists in their research and classification tasks. 
The project aims to streamline the process of pottery analysis, making it faster and more accurate. By using advanced technology, ArchAIDE seeks to enhance archaeological studies and contribute to our understanding of ancient civilizations.

In this repository we have updated the ArchAIDE framework, originally written in Tensorflow 1.x, bringing it to Pytorch, which has a series of features useful for the long-term maintenance of the project.

```
cd reti/training/
python3 archaide_nn_oo.py
```

In the shell you will be asked some questions (which NN? which optimizer? how much dropout? how many epochs? and so on)

Then the training will start.

