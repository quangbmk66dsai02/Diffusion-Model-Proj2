# Diffusion Model for Project

A repository that have 2 conditional models for project 2. 

## Installation

several Python modules are required including pytorch, PIL, and logging. Other packages such as matplotlib, numpy, ... are either dependencies of those 2 packages or already available to the native Python compiler.
Pytorch installation can be tricky, below is a command for CUDA 11.7 (depends on your n

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
streamlit is recommended for a more friendly UI but not required.
```bash
pip install streamlit
```
## Usage

### With streamlit
```bash
streamlit run streamlit_app.py
```
### Without streamlit
1. Open the notebook named DM_project2.ipynb
2. Choose Run all (or similar options) in VSCode (or similar compilers).

DM_project2.ipynb is for the number generation Diffusion Model and the 
DM_project2_64.ipynb is for RGB CIFAR10 generation task. 


