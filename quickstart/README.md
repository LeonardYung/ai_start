# Quickstart

## 1) 安装依赖

```bash
pip install -U langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv
```

## 2) 配置环境变量

在项目根目录创建 `.env`：

```bash
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 3) 运行顺序

1. `python quickstart/01_basic_chain.py`
2. `python quickstart/02_rag_demo.py`
3. `python quickstart/03_structured_output.py`

## 4) 建议记录

- 模型参数：`model`、`temperature`
- RAG 参数：`chunk_size`、`chunk_overlap`、`k`
- 对比不同参数下的回答质量

## 5) 常见问题

- 报错 `OPENAI_API_KEY`：检查 `.env` 是否存在、变量名是否正确。
- RAG 找不到资料：请在 `quickstart/data` 中放置 `.txt` 或 `.md` 文件。
