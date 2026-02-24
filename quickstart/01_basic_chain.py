import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("未检测到 OPENAI_API_KEY，请先在 .env 中配置后再运行。")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个简洁、清晰的AI助教"),
        ("human", "请用{style}风格，用3句话解释：{topic}"),
    ]
)

chain = prompt | llm

resp = chain.invoke({"topic": "LangChain 是什么", "style": "通俗"})
print(resp.content)
