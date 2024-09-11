from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

# To read the api key present in .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize the gemini llm model
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_LLM_API_KEY"], 
                        temperature=0.4)

# Initialize and set the generative ai embeddings
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.environ["GOOGLE_LLM_API_KEY"], 
                                          model="models/embedding-001")

def create_vector_db(source_file) -> object:
    # Load data
    loader = CSVLoader(file_path=source_file, encoding='cp1252')
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)

    return vectordb

def get_qa_chain(vecdb) -> object:
    # Create a retriever for querying the vector database
    retriever = vecdb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this 
    context only. In the answer try to provide as much text as possible from "response" section in the
    source document context without making much changes. If the answer is not found in the context, 
    kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = ( RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) 
                           | PROMPT | llm | StrOutputParser())

    rag_chain_with_source = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})\
        .assign(answer=rag_chain_from_docs)

    return rag_chain_with_source

if __name__ == "__main__":
    source_file = "codebasics_faqs.csv"
    vecdb = create_vector_db(source_file)
    rag_chain = get_qa_chain(vecdb)
    print(rag_chain.invoke("Do you have javascript course?"))