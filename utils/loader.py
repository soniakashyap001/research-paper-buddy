from langchain_community.document_loaders import ArxivLoader, WebBaseLoader, PyMuPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

def load_pdf(path):
    loader = PyMuPDFLoader(path)
    return loader.load()

def load_arxiv(query):
    loader = ArxivLoader(query)
    return loader.load()

def load_web(url):
    loader = WebBaseLoader(url)
    return loader.load()

def get_wikipedia_tool():
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
