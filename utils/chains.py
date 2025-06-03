from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_retrieval_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, doc_chain)
