# Environment Setup Instructions

## Steps Taken

1. **Personal Access to Habrok University Cluster**
    - Gained personal access to the Habrok University cluster.

2. **Load PyTorch Bundle**
    - Loaded the PyTorch bundle with the following command:
        ```bash
        module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
        ```
3. **Create and Activate Conda Environment**
    - Created and activated a conda environment to work within the cluster.
    - Installed necessary packages.
        ```bash
        conda create --n nsaenv
        conda activate nsaenv
        conda install scikit-learn pandas imbalanced-learn numpy matplotlib ipykernel joblib
        conda install -c rapidsai -c conda-forge -c nvidia rapids=24.10 python=3.12 'cuda-version>=12.0,<=12.5'
        conda install cuda-cudart cuda-version=12
        conda install -c conda-forge dask-ml
        ```
    - For LSTM
        ```bash
        conda install keras tensorflow numpy pandas scikit-learn ipykernel imbalanced-learn
        pip install shap
        conda install -c conda-forge notebook
        ```
