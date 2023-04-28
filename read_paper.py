from PyPDF2 import PdfReader
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os, sys
import yaml
import uuid

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
# print(args)

openaikey = os.environ["OPENAI_API_KEY"]


# paperFile = "papers/french-and-romanowicz---2014---whole-mantle-radially-anisotropic-shear-velocity-s.pdf"
paperFile = config['paper_path']

if not os.path.exists(paperFile):
    sys.exit("Error: Paper file does not exist")

########################    MAIN    ##############################
def main():
    docsearch, chain = create_model()

    ## Query the document
    # query = "who are the authors of the article?"
    # query = "explain this article in 10 points"
    query = args.query
    out = query_document(docsearch, chain, query)
    if config['document_output']:
        outdoc = paperFile.replace(".pdf", "_output.txt")
        if config['clear_cache']:
            if os.path.exists(outdoc):
                os.remove(outdoc)
        with open(outdoc, 'a') as f:
            f.write("====================\n")
            f.write("QUERY: {}\n".format(query))
            f.write("OUTPUT: {}\n".format(out))
            f.write("--------------------\n")
    print("--------------------")
    print(out)

def get_size(file_path):
    size = os.path.getsize(file_path)
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def create_model():
    databasedir = "cachedata"
    os.makedirs(databasedir, exist_ok=True)
    # Read the YAML file
    yamldb = "paper_ids.yaml"
    if os.path.exists(yamldb):
        with open(yamldb, 'r') as file:
            yamldata = yaml.safe_load(file)
            if yamldata is None:
                yamldata = {}
    else:
        yamldata = {}

    if paperFile not in yamldata:
        yamldata[paperFile] = {
            'pdfdatafile': "docsearch_{}.pickle".format(str(uuid.uuid4())),
            'chaindatafile': "chain_{}.pickle".format(str(uuid.uuid4()))
            }
        with open(yamldb, 'w') as file:
            yaml.dump(yamldata, file)


    pdfdatafile1 = os.path.join(databasedir, yamldata[paperFile]['pdfdatafile'])
    chaindatafile1 = os.path.join(databasedir, yamldata[paperFile]['chaindatafile'])

    if config['clear_cache']:
        if os.path.exists(pdfdatafile1):
            os.remove(pdfdatafile1)
        if os.path.exists(chaindatafile1):
            os.remove(chaindatafile1)
        print("Cache cleared")

    if not os.path.exists(pdfdatafile1):
        # location of the pdf file/files. 
        reader = PdfReader(paperFile)

        # read data from the file and put them into a variable called raw_text
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text


        # We need to split the text that we read into smaller chunks so that during information 
        # retreival we don't hit the token size limits. 

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = config['chunk_size'],
            chunk_overlap  = config['chunk_overlap'],
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)


        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        docsearch = FAISS.from_texts(texts, embeddings)

        # Save the object to a pickle file
        with open(pdfdatafile1, 'wb') as f:
            pickle.dump(docsearch, f)

        if config['output_stats']:
            print("Number of chunks: {}".format(len(texts)))
            print("Average chunk length: {:.1f}".format(sum([len(t) for t in texts])/len(texts)))
            print("Total length of read: {} characters".format(sum([len(t) for t in texts])))
            print("Total length of original text: {} characters".format(len(raw_text)))
            print("Size of the cache pickle file: {} ".format(get_size(pdfdatafile1)))
    else:
        if config['output_stats']:
            print("Using cached data")

    # Load the object from the pickle file
    with open(pdfdatafile1, 'rb') as f:
        docsearch = pickle.load(f)

    if not os.path.exists(chaindatafile1):
        chain = load_qa_chain(ChatOpenAI(temperature=config['gpt_temperature'], model_name='gpt-3.5-turbo'), chain_type="stuff")
        with open(chaindatafile1, 'wb') as f:
            pickle.dump(chain, f)
    else:
        with open(chaindatafile1, 'rb') as f:
            chain = pickle.load(f)
    return docsearch, chain

def query_document(docsearch, chain, query):
    docs = docsearch.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

if __name__ == "__main__":
    main()