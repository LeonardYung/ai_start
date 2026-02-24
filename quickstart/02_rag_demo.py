import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("未检测到 OPENAI_API_KEY，请先在 .env 中配置后再运行。")

text_loader = DirectoryLoader("quickstart/data", glob="**/*.txt", loader_cls=TextLoader)
md_loader = DirectoryLoader("quickstart/data", glob="**/*.md", loader_cls=TextLoader)
docs = text_loader.load() + md_loader.load()

if not docs:
    raise ValueError("quickstart/data 目录下未找到 .txt 或 .md 文档，请先添加学习资料。")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = splitter.split_documents(docs)

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(splits, emb)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """你是问答助手。请仅基于上下文回答；如果上下文不足，请明确说不知道。

上下文：
{context}

问题：{question}
"""
)

question = "请总结这些资料中最关键的3点"
retrieved_docs = retriever.invoke(question)
context = "\n\n".join([d.page_content for d in retrieved_docs])

chain = prompt | llm
resp = chain.invoke({"context": context, "question": question})

print("=== 回答 ===")
print(resp.content)
print("\n=== 引用片段 ===")
for i, d in enumerate(retrieved_docs, 1):
    print(f"[{i}] {d.page_content[:120]}...")
