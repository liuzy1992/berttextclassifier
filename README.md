# BERTtextclassifier
**BERTtextclassifier** is a text classifier based on pre-trained BERT model.

## Requirement
python3  
pytorch==1.6.0  
torchtext==0.8.0  
pandas  
matplotlib  
transformers==3.0.0  
sklearn  
seaborn  

## Usage
```shell
./BERTtextclassifier.sh [-i infile]
                        [-m model_path]
                        <-d data_outdir>
                        <-o model_outdir>
                        <-l max_length>
                        <-b batch_size>
                        <-n num_epochs>
                        <-r learning_rate>
```
**-i**: filename of raw input text. Input text shoud be in TSV format with 4 colomns as follows:  
        ID<TAB>title<TAB>content<TAB>label  
        labels should be intergers.  
**-m**: directory of pre-trained BERT model.  
**-d**: directory for output preprocessed text data. Default=./data_preprocessed  
**-o**: directory for output post-trained model. Default=./model  
**-l**: max length of sequences. Default=150  
**-b**: batch size for training, valid and test data. Default=4  
**-n**: number of epochs to use. Default=5  
**-r**: learning rate of model. Default=2e-5  

## Pre-trained Model
Pre-trained BERT model should be download first for training.  
**Example**  
To download bioBERT pretrained model. use folloing command:  
```shell
git lfs install
git clone https://huggingface.co/dmis-lab/biobert-v1.1
```
Then pre-trained bioBERT model will be download in ./biobert-v1.1.

## Example
With downloaded bioBERT model, use following command to test:
```shell
./BERTtextclassifier.sh -i testdata/test.tsv -m biobert-v1.1
```
