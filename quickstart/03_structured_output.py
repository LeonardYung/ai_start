import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("未检测到 OPENAI_API_KEY，请先在 .env 中配置后再运行。")


class Summary(BaseModel):
    title: str = Field(description="主题标题")
    key_points: List[str] = Field(description="关键点列表")
    action_items: List[str] = Field(description="下一步行动")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(Summary)

result = structured_llm.invoke("总结 LangChain 快速入门路线，并给出3个可执行行动")
print(result.model_dump())
