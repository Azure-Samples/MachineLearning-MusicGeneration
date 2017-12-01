
# Music Generation with Azure Machine Learning

Sequence-to-Sequence model using multi-layered LSTM for music generation. For more detailed walkthrough see: add blog link
## Setup compute environment

### Setup remote VM as execution target
```
az ml computetarget attach --name "my_dsvm" --address "my_dsvm_ip_address" --username "my_name" --password "my_password" --type remotedocker
```
### Configure my_dsvm.compute
```
baseDockerImage: microsoft/mmlspark:plus-gpu-0.7.91
nvidiaDocker: true
```
### Configure my_dsvm.runconfig
To push models to Azure Blob Storage, add your storage account details to your .runconfig file:

```
EnvironmentVariables:
  "STORAGE_ACCOUNT_NAME": "<YOUR_AZURE_STORAGE_ACCOUNT_NAME>"
  "STORAGE_ACCOUNT_KEY": "<YOUR_AZURE_STORAGE_ACCOUNT_KEY>"
Framework: Python
```

For more info on Azure ML Workbench compute targets see [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-create-dsvm-hdi).

## Train

To train your own model using a DSVM compte target

### Prepare compute environment

```
az ml experiment -c prepare my_dsvm
```

### Run the experiment

```
az ml experiment submit -c my_dsvm MusicGeneration/train.py
```

## Generate Music (Predict)

```
az ml experiment submit -c my_dsvm MusicGeneration/score.py
```

## Data Credit

The dataset used for the experiments is available at (http://www.feelyoursound.com/scale-chords/)
