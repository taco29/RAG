import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from retrieve import Retriever
from langchain_core.prompts import ChatPromptTemplate

API_KEY = ""
os.environ["GOOGLE_API_KEY"] = API_KEY

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Bạn là một trợ giảng Machine Learning.

Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên thông tin có trong CONTEXT.
KHÔNG được sử dụng kiến thức bên ngoài.
KHÔNG suy đoán.
KHÔNG bịa thông tin.

Nếu trong CONTEXT không có thông tin để trả lời câu hỏi,
hãy trả lời chính xác:
"Không có thông tin trong tài liệu được cung cấp."

====================
CONTEXT:
{context}
====================

QUESTION:
{question}

Yêu cầu câu trả lời:
- Viết bằng tiếng Việt
- Ngắn gọn, rõ ràng
- Có cấu trúc (gạch đầu dòng hoặc từng ý)
- Dùng thuật ngữ chính xác
- Không thêm ví dụ ngoài tài liệu

CÂU TRẢ LỜI:
""")

query = "Markov Networks là gì? Chúng được ứng dụng như thế nào trong Machine Learning?"

path = r"C:\Users\Admin\Code\Chatbot\RAG\data.txt"
retriever_instance = Retriever(path, k=3).get_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=retriever_instance,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
result = qa_chain({"query": query})

print("ANSWER:")
print(result["result"])

print("\nSOURCES:")
for doc in result["source_documents"]:
    print("-", doc.metadata)



print("\n--- CHUNKS IN THE DOCUMENT ---")
docs = Retriever(path, k=3).get_chunks()
for i, doc in enumerate(docs):
    print(f"Chunk {i+1}:")
    print(doc.page_content)
    print("--------------------")   
