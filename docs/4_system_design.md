# System Analysis & Design

## System Architecture
The system follows a Retrieval-Augmented Generation (RAG) pipeline:

User → API → Retriever → Vector Database → Language Model → Response

---

## Components

### 1. Data Layer
- Medical datasets  
- FAQs and documentation  

---

### 2. Vector Database
- FAISS or Pinecone for similarity search  

---

### 3. RAG Pipeline
- **Retriever:** Fetches relevant documents  
- **Generator (LLM):** Produces final answer  

---

### 4. Backend
- FastAPI or Flask for API development  

---

### 5. Deployment
- Azure App Service or Azure Machine Learning  

---

## System Flow
1. User sends a query  
2. API processes the request  
3. Retriever searches in vector database  
4. Relevant documents are passed to the LLM  
5. LLM generates the final answer  
6. Response is returned to the user  