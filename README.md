# Article Research Tool



## Project Overview



An AI-powered tool that allows users to input article links and ask questions related to the article, providing answers using advanced NLP and generative AI technologies. Built with LangChain, Hugging Face, GooglePalm, and Streamlit.



## Technologies Used



- **LangChain:** UnstructuredURLLoader, RecursiveCharacterTextSplitter
- **HuggingFaceEmbeddings**
- **RetrievalQA**
- **GooglePalm LLM:** temperature=0.2, max_tokens=500
- **Streamlit:** For the user interface



## Features



- **Document Reading:** Uses UnstructuredURLLoader for loading articles.
- **Text Splitting:** Utilizes RecursiveCharacterTextSplitter for better text processing.
- **Text Embeddings:** Created using HuggingFaceEmbeddings.
- **Efficient Querying:** Developed a vector database with RetrievalQA.
- **AI Answer Generation:** Employed GooglePalm LLM for generating answers.
- **User Interface:** Designed with Streamlit for a seamless user experience.



## Setup Instructions



1. **Clone the repository:**
   git clone https://github.com/yourusername/ArticleResearchTool.git
   cd ArticleResearchTool



2. Install dependencies:
   pip install streamlit langchain huggingface_hub google-palm python-dotenv Streamlit



3. Acquire an api key through makersuite.google.com and put it in .env file
   GOOGLE_API_KEY="your_api_key_here"



## Usage/Examples



1. Run the Streamlit app by executing:
   streamlit run main.py



2.The web app will open in your browser.



- On the sidebar, you can input URLs directly.



- Initiate the data loading and processing by clicking "Process URLs."



- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.



- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.



- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles



## Project Structure



- main.py: The main Streamlit application script.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.



   