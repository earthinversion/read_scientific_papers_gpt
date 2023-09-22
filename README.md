## Language Model for Reading Research Papers
This script reads a PDF file and answers questions about it. The script takes two command-line arguments: `--configfile` which specifies the path to a YAML configuration file, and `--query` which is the query to ask about the PDF file.

## Description
- `read_paper.py`: It uses the `load_qa_chain` that provides the most generic interface for answering questions. It loads a chain that you can do QA for your input documents. It uses ALL of the text in the documents.
- `conversational.py`: This method retains the memory of the previous conversation. It can use the memory to improve the answer.

## Installation
- Create a python virtual environment and install these libraries
```
python3 -m venv venv
source venv/bin/activate
pip install langchain
pip install openai
pip install PyPDF2
pip install faiss-cpu
pip install tiktoken
pip install pyyaml
pip install pypdf
pip install chromadb
```

## Run
- Rename 'config.yaml.example' to 'config.yaml', check configurations inside
- Export your OpenAI api in your os environment. You can go to the openai platform to get your api: https://platform.openai.com/account/api-keys
- This script uses gpt-3.5-turbo model
- For a list of other models: https://platform.openai.com/docs/models/overview

## Examples

### Who are the authors of this article?
```
python read_paper.py -c config.yaml -q "who are the authors of the article?"
```

```
python conversational.py -c config_conversational.yaml -q "What is the Farallon slab?"
```

### Summary of this article?
```
python read_paper.py -c config.yaml -q "write a summary of this article"
```
