# dslab-rag-retrieval

# Pre Requisites

This environment has been tested in Win11 with Py 3.10

Install Virtualenv package:
```bash
pip install virtualenv
```

Create a virtual environment
```bash
python venv .venv
```

Activate the Virtual Environment,
following the specific instructions for your OS,
for example, with Win/Powershell: `.\.venv\Scripts\activate`


Install dependencies
```bash
pip install requirements.txt
```

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

Then access


