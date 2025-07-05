# Chat with your documents

This project implements a Retrieval-Augmented Generation (RAG) chatbot, combining large language models (LLMs) with your own data to enable advanced, context-aware chat capabilities.

## **Project Structure**

```
RAG_LLM_NOTES/
├── __pycache__/
├── .gradio/
├── chat_logs/
├── data/
│   └── laracourses/
├── logs/
├── .env
├── .gitignore
├── AIprompts.json
├── chatutils.py
├── config.yaml
├── db_vector_utils.py
├── documents_util.py
├── helperUtil.py
├── rag_llm_ai_notes.ipynb
├── readme.md
└── requirement.txt
```

## **Getting Started**

### **1. Install Requirements**

```bash
pip install -r requirement.txt
```

### **2. Set Up Environment Variables**

Edit the `.env` file and add your required API keys and variables:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### **3. Configure Project Settings**

Update `config.yaml` and other configuration files to suit your environment and requirements.

### **4. Run the Main Notebook**

1. Open `rag_llm_ai_notes.ipynb` in Jupyter Notebook or JupyterLab
2. Execute each cell step by step as instructed in the notebook

## **Main Files**

- **`rag_llm_ai_notes.ipynb`** — Main notebook to run, index data, and chat with your RAG assistant
- **`chatutils.py`** — Chat utility functions
- **`db_vector_utils.py`** — Database and vector operations
- **`documents_util.py`** — Document processing utilities
- **`helperUtil.py`** — General helper functions
- **`config.yaml`** — Project configuration
- **`AIprompts.json`** — Custom AI prompt templates
- **`.env`** — Environment variables
- **`requirement.txt`** — List of Python dependencies

## **Notes**

- Logs are saved in the `chat_logs/` and `logs/` directories
- Place your data files (for example, course material) inside the `data/laracourses/` directory
