# LangChain 3小时快速上手：实操导向版

> 目标：**3小时内建立 LangChain 全局认知 + 做出一个可运行的小应用**（带基础 RAG）。

---

## 你会在 3 小时内得到什么

- 知道 LangChain 在 LLM 应用里的角色（模型编排层）。
- 会写最小链路：`Prompt -> Model -> Output`。
- 会做一个最小 RAG 问答（本地文档检索 + 回答）。
- 有一份可继续扩展到 Agent/LangGraph 的工程骨架。

---

## 一、3 小时计划（按分钟）

## 0:00 - 0:20（20分钟）环境与心智模型

### 学习目标
- 建立 LangChain 的全局图：
  - Model（模型）
  - Prompt（提示模板）
  - Chain/Runnable（编排）
  - Retriever（检索）
  - Memory/Agent（后续进阶）

### 实操
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv
```

新建 `.env`：
```bash
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 0:20 - 1:00（40分钟）最小可运行链路

### 学习目标
- 体验 LCEL（`|` 管道式拼接）。
- 学会参数化 Prompt。

### Step 1：创建 `quickstart/01_basic_chain.py`

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁、清晰的AI助教"),
    ("human", "请用{style}风格，用3句话解释：{topic}")
])

chain = prompt | llm

resp = chain.invoke({"topic": "LangChain 是什么", "style": "通俗"})
print(resp.content)
```

### Step 2：运行
```bash
python quickstart/01_basic_chain.py
```

### Step 3：立即练习（5分钟）
- 把 `style` 改成 `专业`、`面试回答`、`给小白`。
- 把 `topic` 改成 `RAG`、`Agent`、`向量数据库`。

---

## 1:00 - 1:45（45分钟）做最小 RAG（核心）

### 学习目标
- 理解 RAG 主链路：加载文档 → 切分 → 向量化 → 检索 → 回答。

### Step 1：准备资料
创建目录并放入 1~3 个 txt/md 文档（脚本会同时读取 `.txt` 和 `.md`）：
```bash
mkdir -p quickstart/data
```

### Step 2：创建 `quickstart/02_rag_demo.py`

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1) 加载文档
loader = DirectoryLoader("quickstart/data", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# 2) 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = splitter.split_documents(docs)

# 3) 向量化+建库
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(splits, emb)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4) 生成
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
```

### Step 3：运行
```bash
python quickstart/02_rag_demo.py
```

### Step 4：立即练习（10分钟）
- 调 `k=2/4/6` 比较回答稳定性。
- 调 `chunk_size=300/800` 比较召回质量。

---

## 1:45 - 2:20（35分钟）增加结构化输出（工程必备）

### 学习目标
- 把模型回复变成结构化字段，方便存库/API返回。

### Step 1：创建 `quickstart/03_structured_output.py`

```python
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Summary(BaseModel):
    title: str = Field(description="主题标题")
    key_points: List[str] = Field(description="关键点列表")
    action_items: List[str] = Field(description="下一步行动")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(Summary)

result = structured_llm.invoke("总结 LangChain 快速入门路线，并给出3个可执行行动")
print(result.model_dump())
```

### Step 2：运行
```bash
python quickstart/03_structured_output.py
```

---

## 2:20 - 3:00（40分钟）串成一个小项目 + 复盘

### 学习目标
- 把前面 3 个脚本连成“可交付”的雏形。

### 实施指南（一步一步）

1. 创建项目结构：
   ```bash
   mkdir -p quickstart/{data,outputs}
   ```
2. 运行 `01_basic_chain.py`，把结果保存到 `quickstart/outputs/basic.txt`。
3. 运行 `02_rag_demo.py`，把回答和引用保存到 `quickstart/outputs/rag.txt`。
4. 运行 `03_structured_output.py`，把 JSON 保存到 `quickstart/outputs/summary.json`。
5. 写一个 `quickstart/README.md`，记录：
   - 你的模型参数（model/temperature）
   - 你的 RAG 参数（chunk_size/chunk_overlap/k）
   - 你观察到的最优组合

### 3小时结束时的验收标准
- 你能解释 LangChain 的 5 个组件各做什么。
- 你能独立运行 3 个脚本并改参数。
- 你有一个最小 RAG demo 能回答你自己的资料。

---

## 二、完成 3 小时后的下一步（可选）

### 24小时内（巩固）
- 加一个工具调用：`calc(expression)`。
- 给 RAG 增加“回答必须给出处编号”的约束。

### 3天内（进阶）
- 用 FastAPI 封装 `/chat` 和 `/rag/query`。
- 引入 LangGraph，把流程做成“检索 -> 回答 -> 自检”三节点。

---

## 三、常见报错与排查

- `OPENAI_API_KEY missing`：检查 `.env` 是否加载、变量名是否正确。
- `No module named ...`：确认在 `.venv` 中执行并重新 `pip install`。
- RAG 检索不准：先调 `chunk_size` 和 `k`，再考虑更换 embedding 模型。

---

## 四、你现在就可以执行（复制即用）

```bash
source .venv/bin/activate
python quickstart/01_basic_chain.py
python quickstart/02_rag_demo.py
python quickstart/03_structured_output.py
```

如果你愿意，我下一步可以继续给你：
- **“第2个3小时训练营”**（Tool Calling + 简易 Agent）
- 并且按你当前代码逐文件做 review。
