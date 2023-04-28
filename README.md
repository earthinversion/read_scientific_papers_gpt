## Language Model for Reading Research Papers
https://platform.openai.com/docs/models/overview
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
- Export your OpenAI api in your os environment

## Examples

### Who are the authors of this article?
```
python read_paper.py -c config.yaml -q "who are the authors of the article?"
```

### Summary of this article?
```
python read_paper.py -c config.yaml -q "write a summary of this article"
```
