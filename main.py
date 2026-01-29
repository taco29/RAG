import os
from retrieve import Retriever
from gen import AnswerGenerator

API_KEY = ""
os.environ["GOOGLE_API_KEY"] = API_KEY

def main():
    path = r"C:\Users\Admin\Code\Chatbot\RAG\data.txt"
    question = "Bagging là gì?"

    retriever = Retriever(path, k=3).get_retriever()
    result = AnswerGenerator().answer(retriever, question)
    print("ANSWER:")
    print(result["result"])

    print("\nSOURCES:")
    for doc in result["source_documents"]:
        print("-", doc.metadata)
        print(doc.page_content[:200], "...\n")

    print("\n--- CHUNKS IN THE DOCUMENT ---")
    docs = Retriever(path, k=3).get_chunks()
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:")
        print(doc.page_content)
    print("--------------------")   
if __name__ == "__main__":
    main()