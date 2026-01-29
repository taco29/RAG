import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

PROMT = """
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
"""

class AnswerGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.prompt = self._build_prompt()

    def _build_prompt(self):
        return ChatPromptTemplate.from_template(PROMT)
    
    def create_chain(self, retriever):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            #chain_type_kwargs={"prompt": self.prompt}
        )
    
    def answer(self, retriever, question: str):
        qa_chain = self.create_chain(retriever)
        return qa_chain({"query": question})