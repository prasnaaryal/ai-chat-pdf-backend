import os
import json
import uuid
from io import BytesIO
from typing import List

from api import req
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from langchain.schema import Document


from .db import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI app
from .db import SessionLocal, engine
from .models import Base, ChatHistory, Chat
from fastapi.middleware.cors import CORSMiddleware

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Dependency to get the SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize Chroma DB client
client = chromadb.PersistentClient(path="./chroma_db")
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)

# Initialize LangChain components
embeddings_retriever = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,
    google_api_key=google_api_key
)
template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
"""
prompt = PromptTemplate.from_template(template)

# Create or get Chroma collection
collection_name = "forensic"
collection = client.get_or_create_collection(name=collection_name, embedding_function=google_ef)

def get_pdf_text(pdf_docs: List[BytesIO]) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_docs_to_add_vectorstore(pages: List[str], file: str):
    documents = []
    ids = []
    metadatas = []
    embeddings = []

    for page in pages:
        emb = google_ef([page])
        embeddings.append(emb[0])
        metadatas.append({"page": "page", "filename": file})
        ids.append(uuid.uuid1().hex)
        documents.append(page)
    return documents, ids, metadatas, embeddings


@app.post("/chat/")
async def new_chat(db: Session = Depends(get_db), files: List[UploadFile] = File(...)):
    try:
        # Read the PDF files and extract text
        pdf_files = [BytesIO(await file.read()) for file in files]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)

        # Add data to Chroma DB
        documents, ids, metadatas, embeddings = get_docs_to_add_vectorstore(text_chunks, files[0].filename)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Create and save a new ChatHistory record
        chat_history = ChatHistory(title=files[0].filename, context=raw_text)
        db.add(chat_history)
        db.commit()
        db.refresh(chat_history)

        # Serialize the ChatHistory object and return it as a JSON response
        chat_history_data = jsonable_encoder(chat_history)
        return JSONResponse(content=chat_history_data)

    except Exception as e:
        # Handle exceptions and return an error response
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Ensure the session is closed
        db.close()

@app.get("/chats/")
async def get_chats(db: Session = Depends(get_db)):
    try:
        # Fetch all chat history records from the database
        chat_histories = db.query(ChatHistory).all()

        # Serialize the chat histories
        chat_histories_data = jsonable_encoder(chat_histories)

        # Return the chat histories as a JSON response
        return JSONResponse(content=chat_histories_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/conversation/")
async def talk_with_gpt(question: req.ConversationRequest, db: Session = Depends(get_db)):
    try:
        # Retrieve the chat history using chat_id
        chat_history = db.query(ChatHistory).filter(ChatHistory.id == question.chat_id).first()
        if not chat_history:
            raise HTTPException(status_code=404, detail="ChatHistory not found")

        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        documents = [Document(page_content=chunk) for chunk in get_text_chunks(chat_history.context)]

        response = chain({"input_documents": documents, "question":question.question}, return_only_outputs=True)
        answer = response.get("output_text", "No answer available.")

        chat = Chat(chat_history_id=question.chat_id, question=question.question, answer = answer)
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return JSONResponse(content=jsonable_encoder(chat))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()

@app.get("/conversation/")
async def get_all_chats(chat_id: int , db: Session = Depends(get_db)):
    try:
        # Retrieve the chat history using chat_id
        chats = db.query(Chat).filter(Chat.chat_history_id == chat_id).all()
        chat_lists = jsonable_encoder(chats)
        return JSONResponse(content=chat_lists)


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)