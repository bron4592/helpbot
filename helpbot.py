'''
Following this blog post: https://www.paepper.com/blog/posts/build-q-and-a-bot-of-your-website-using-langchain/
and using streamlit as the framework

TO RUN:
- pip install packages
- Enter your OpenAPI key on line 25
- from terminal run: streamlit run helpbot.py
'''

import streamlit as st 
import xmltodict
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
import faiss

# enter your openAI apikey
apikey = ''

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

# query all pages in www.facebook.com/business domain
r = requests.get("https://www.facebook.com/business/sitemap.xml")
xml = r.text
raw = xmltodict.parse(xml)

# filter for help articles
pages = []
for info in raw['urlset']['url']:
    url = info['loc']
    # I set this to query one page as a proof of concept, 
    # remove 523975231703117 to query all help center articles (takes forever)
    # this article is about branded content
    if 'https://www.facebook.com/business/help/' in url:
        pages.append({'text': extract_text_from(url), 'source': url})

# split each page’s content into a number of documents
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs, metadatas = [], []
for page in pages:
    splits = text_splitter.split_text(page['text'])
    docs.extend(splits)
    metadatas.extend([{"source": page['source']}] * len(splits))
    print(f"Split {page['source']} into {len(splits)} chunks")

# vector store of embeddings
store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=apikey), metadatas=metadatas)
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

# app framework
st.title('Facebook Business Help Center Q&A')
prompt = st.text_input('Ask your question here') 

# llm
with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
chain = VectorDBQAWithSourcesChain.from_llm(
            llm=OpenAI(openai_api_key=apikey,temperature=0), vectorstore=store)

# magic???
if prompt:
    st.write()
    result = chain({"question": prompt})
    st.write(result['answer']) 
    st.write(result['sources'])