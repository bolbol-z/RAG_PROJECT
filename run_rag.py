from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.ollama import OllamaEmbeddings
from pathlib import Path
import os

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set. Run: export GOOGLE_API_KEY=your_key")

CHROMA_PATH = Path("chroma_db")

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

PROMPT_TEMPLATE = """
Answer the question based only on the following context. 
Formulate the answer with extreme accuracy based on the context, without adding or removing information:

Context:
{context}

---
Question: {question}
"""

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=str(CHROMA_PATH), 
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)
    response = model.invoke(prompt)

    if isinstance(response.content, list):
        response_text = "".join([item.get("text", "") for item in response.content if isinstance(item, dict)])
    else:
        response_text = str(response.content)

    print("\n" + "="*50)
    print(" Answer ")
    print("="*50)
    print(response_text)
    
    print("\n" + "="*50)
    print(" Sources ")
    print("="*50)
    sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
    for source in sources:
        print(f" • {source}")
    print("="*50 + "\n")
    
    return response_text

if __name__ == "__main__":
    question = "Give me the names of the repo from GitHub that we will use to build the project and explain to me what we need and what we will take from them exactly ? "
    print(f"\nSearching and analyzing for question: {question}")
    query_rag(question)