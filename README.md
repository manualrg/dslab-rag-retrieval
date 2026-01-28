# dslab-rag-retrieval

# Pre Requisites

Create a virtual environment
```bash
python -m venv .venv
```

Activate the Virtual Environment,
following the specific instructions for your OS,
for example, with Win/Powershell: `.\.venv\Scripts\activate`


Install dependencies
```bash
pip install -r requirements.txt
```

This virtual environment has been built using Win11 and Python 3.10.9, 3.11  

As project is based in notebooks, three popular approaches can be followed:
* Install jupyter notebook: `pip install jupyter`
* Install jupyterlab: `pip install jupyterlab`
* Use VSCode notebooks extension: [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) 
and install ipykernel with `pip install ipykernel`


## Enviroment Variables
Create a `.env` file and populate the following environment variables:
```
OPENAI_API_KEY=
QDRANT_API_KEY=
QDRANTL_URL=
MISTRAL_API_KEY=
```


## HuggingFace dataset
Create and logging with your HF accounnt to download the dataset.
[Retrieval-Augmented-Generation and Queston-Answering in Spanish (RagQuAS) Dataset](https://huggingface.co/datasets/IIC/RagQuAS)
In your broser, insert the link above to get to the dataset page, then request access.

In your terminal:
1. Check that HF auth library is working: `hf --help`
2. Generate a HF token [User access tokens](https://huggingface.co/docs/hub/security-tokens)
3. Use the library to loggin: `hf auth login` and insert the token when prompted 


# Set up in Colab

1. In colab notebook cell, download the repository
```python
!git clone `repo-url`
```
This will create a folder named `dslab-rag-retrieval` You can see it in  the navegator.
Therefore, you can get access to the data/ and /src folders. You must manually move those folders to the root (outside dslab-docai),
so your imports and data read/write point to the proper paths

2. Add your secrets in colab

3. Install dependencies
Add the following cell to your current notebook to install dependencies
```python
%%capture
import os, sys
if "COLAB_" in "".join(os.environ.keys()):
  print(f"Running in colab")
  !pip install docling==2.43.0 docling-core[chunking-openai]==2.43.0 mistralai==1.9.10 langchain-text-splitters==0.3.11 langchain-docling==1.0.0 langchain-experimental==0.3.4 
else:                                                                                                                                                                                     
    print(f"Running in {sys.executable}")
```
Remember to start with `00-setup.ipynb`

4. Modifications:
Docling models are downloaded at first call, not by a specific sentence    
Rmove explicit docling artifacts loading   
```python
pipeline_options = PdfPipelineOptions(
    # artifacts_path=path_artifacts.as_posix(),
    )
```
