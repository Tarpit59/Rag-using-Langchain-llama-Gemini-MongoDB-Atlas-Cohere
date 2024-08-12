
# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF files and ask questions based on the content of those files. The chatbot is built using Python, Flask, HTML, CSS, and JavaScript, Langchain framework, with MongoDB Atlas serving as the vector database for storing and retrieving document embeddings.

The chatbot offers two Large Language Model (LLM) options:

    1. Gemini-1.5-Flash-001-Tuning
    2. Llama3-8b-8192

For embedding generation, the project utilizes the GoogleGenerativeAIEmbeddings's "models/embedding-001" model. Additionally, the Cohere model "rerank-english-v3.0" is employed for reranking the retrieved results to improve the relevance of answers.

This RAG chatbot is designed to be easily integrated into various applications where users need to query large sets of documents and receive precise and contextually relevant answers.


## API Reference

#### Upload Files

```http
  POST /upload
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `files` | `file[]` | **Required**. The PDF files to be uploaded. |

#### Query RAG

```http
    POST /query
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `model`   | `string` | **Required**. The model to use ('google' or 'llama'). |
| `query`   | `string` | **Required**. The question you want to ask the chatbot. |



## Tech Stack

**Client**: HTML, CSS, JavaScript

**Server**: Python, Flask

**Database**: MongoDB Atlas (Vector Database)

**Embeddings**: GoogleGenerativeAIEmbeddings (models/embedding-001)

**Reranking**: Cohere Rerank Model (rerank-english-v3.0)

**LLM Models**: Gemini-1.5-Flash-001-Tuning, Llama3-8b-8192


## Features

- Multiple PDF Uploads: Allows users to upload and process multiple PDF files simultaneously.
- Customizable Query Models: Offers two Large Language Model (LLM) options—Gemini-1.5-Flash-001-Tuning and Llama3-8b-8192—for querying.
- Advanced Embedding and Reranking: Utilizes GoogleGenerativeAIEmbeddings for creating embeddings and the Cohere Rerank model for improving query relevance.
- Responsive Web Interface: Built with HTML, CSS, JavaScript, and Flask, ensuring a user-friendly experience across different devices.
- Efficient Vector Search: Powered by MongoDB Atlas for fast and accurate similarity searches within large document collections.


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file


`MONGODB_ATLAS_CLUSTER_URI`: Your MongoDB Atlas Cluster URI

`GOOGLE_API_KEY`: API key for Google Generative AI

`GROQ_API_KEY`: API key for Groq

`COHERE_API_KEY`: API key for Cohere


## Deployment

1. Create a Cluster on MongoDB Atlas:
- Sign in to your MongoDB Atlas account.
- Create a new cluster.
- Once your cluster is set up, create a new database and collection that will be used for vector search.

2. Create a Vector Search Index:
- Navigate to your MongoDB Atlas cluster and select your database and collection.
- Create an index with the following configuration:
```bash
  {
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "source",
      "type": "filter"
    }
  ]
}
```
3. Allow Network Access:
- Go to the "Network Access" section of MongoDB Atlas.
- Add your current IP address to the IP whitelist to allow access to the database.

4. Set Up Environment Variables:
- Create a .env file in the root of your project.
- Add the following environment variables:
```bash
MONGODB_ATLAS_CLUSTER_URI=your_mongodb_atlas_uri
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```
5. Install Dependencies:
- Make sure you have all necessary dependencies installed by running:
```bash
pip install -r requirements.txt
```
6. Run the Application:
- Start the Flask application by running:
```bash
python app.py
```

## Screenshots

![App Screenshot](https://github.com/Tarpit59/Rag-using-Langchain-llama-Gemini-MongoDB-Atlas-Cohere/blob/master/base/static/Screenshot/UI_Screenshot.png)


## Acknowledgements

 - [LangChain RAG Tutorial](https://python.langchain.com/v0.2/docs/tutorials/rag/)
 - [MongoDB Atlas Vector Search Integration](https://python.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/)
 - [Leveraging MongoDB Atlas Vector Search](https://www.mongodb.com/developer/products/atlas/leveraging-mongodb-atlas-vector-search-langchain/)
 - [Cohere ReRank with LangChain](https://docs.cohere.com/docs/rerank-on-langchain#:~:text=for%20more%20information.-,Cohere%20ReRank%20with%20LangChain,retrievers%2C%20embeddings%2C%20and%20RAG.)

## 🚀 About Me
My name is Tarpit, and I completed my B.Tech in Computer Engineering with a specialization in AI and Machine Learning in 2024. I'm passionate about developing intelligent systems and leveraging AI to solve complex problems. This project is a reflection of my interest in combining cutting-edge technologies like RAG, MongoDB Atlas, and advanced LLMs to create powerful and user-friendly applications.


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tarpit-patel)

