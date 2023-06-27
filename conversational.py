## Following this article: https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a
## to retrieve relevant text chunks first and only use the relevant text chunks in the language model.
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
import os, sys
import yaml
import pickle

import argparse


# Create the parser
parser = argparse.ArgumentParser(description='Read a paper and answer questions about it.')

# Add arguments
parser.add_argument('-c','--configfile', type=str, help='the path to the paper')
parser.add_argument('-q','--query', type=str, help='the query to ask about the paper')


# Parse the arguments
args = parser.parse_args()

with open(args.configfile, 'r') as f:
    config = yaml.safe_load(f)

# load document
from langchain.document_loaders import PyPDFLoader
paperFile = config['paper_path']

## check if the paper file exists
if not os.path.exists(paperFile):
    sys.exit("Error: Paper file does not exist")

## check if the paper file is a PDF
if not paperFile.endswith(".pdf"):
    print("Error: Paper file must be a PDF")
    sys.exit(1)


def main():
    databasedir = "cachedata" ## directory to store the cache files
    if not os.path.exists(databasedir):
        os.makedirs(databasedir)

    loader = PyPDFLoader(paperFile)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1}, max_tokens_limit=4097)

    qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)

    chat_history_file = os.path.join(databasedir, "chat_history_data.pkl")
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'rb') as f:
            chat_history = pickle.load(f)
    else:
        chat_history = []

    # query = "Who are the authors of this paper?"
    result = qa({"question": args.query, "chat_history": chat_history})
    # print(result)

    if len(chat_history) > 0:
        for val in chat_history:
            print("--> {}: {}".format(val[0], val[1]))

    print("--> {}: {}".format(args.query, result['answer']))

    # Save the object to a pickle file
    queryIndex = checkExistingQuery(chat_history, args.query)
    if queryIndex:
        chat_history[queryIndex] = (args.query, result['answer'])
    else:
        chat_history.append((args.query, result['answer']))
    with open(chat_history_file, 'wb') as f:
        pickle.dump(chat_history, f)


def checkExistingQuery(history, query):
    for ival, val in enumerate(history):
        if query in val[0]:
            return ival
    return False

if __name__ == "__main__":
    main()