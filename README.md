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
- Export your OpenAI api in your os environment

```
python read_paper.py -pp "papers/french-and-romanowicz---2014---whole-mantle-radially-anisotropic-shear-velocity-s.pdf" -q "who are the authors of the article?"
```
