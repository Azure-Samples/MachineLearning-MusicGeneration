
# Music Generation with Azure Machine Learning

> **NOTE** This content is no longer maintained. Visit the [Azure Machine Learning Notebook](https://github.com/Azure/MachineLearningNotebooks) project for sample Jupyter notebooks for ML and deep learning with Azure Machine Learning.

Sequence-to-Sequence model using multi-layered LSTM for music generation. For more detailed walkthrough see: [blog](https://blogs.technet.microsoft.com/machinelearning/2017/12/06/music-generation-with-azure-machine-learning/)

## Prerequisites

The prerequisites to run this example are as follows:

1. Make sure that you have properly installed [Azure Machine Learning Workbench](https://docs.microsoft.com/en-us/azure/machine-learning/preview/overview-what-is-azure-ml) by following the [Install and create Quickstart](https://docs.microsoft.com/en-us/azure/machine-learning/preview/quickstart-installation)

2. This example could be run on any compute context. However, it is recommended to run it on a GPU machine to accelerate the training process.

3. Access to an Azure Blob Storage Account. See how to create and manage your storage account [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-create-storage-account?toc=%2fazure%2fstorage%2fblobs%2ftoc.json)

## Create a new Workbench project

1. Clone this repo to your local machine to /MachineLearning-MusicGeneration
2. Open Azure Machine Learning Workbench
3. On the Projects page, click the + sign and select Add Existing Folder as Project
4. Delete the .git folder in the cloned repo as Azure Machine Learning Workbench currently cannot import projects that contain a git repo
4. In the Add Existing Folder as Project pane, set the project directory to the location where this repo has been cloned and fill in the information for your new project
5. Click Create

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
## Listen to your own music!

The song generated in the previous step will be saved in your Blob Storage conatiner. You can listen to the song by downloading the .mid file and playing it using any standard media player like Windows Media Player for example.

## Data Credit

The dataset used for the experiments is available at (http://www.feelyoursound.com/scale-chords/)
