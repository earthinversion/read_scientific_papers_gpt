## Language Model for Reading Research Papers
## Installation
- Create a python virtual environment and install these libraries
```
pip install langchain
pip install openai
pip install PyPDF2
pip install faiss-cpu
pip install tiktoken
pip install pyyaml
```

## Run
- Export your OpenAI api in your os environment. You can go to the openai platform to get your api: https://platform.openai.com/docs/models/overview

## Examples

### Who are the authors of this article?
```
python read_paper.py -c config.yaml -q "who are the authors of the article?"
```

### Summary of this article?
```
python read_paper.py -c config.yaml -q "write a summary of this article"
```
