import os
import json
import uuid
import boto3
from io import BytesIO
from typing import List
from typing import Dict

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
from fastapi.middleware.cors import CORSMiddleware
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pydantic import BaseModel
from fastapi import Request, HTTPException, Depends, Query

from db import SessionLocal, engine
from models import Base, ChatHistory, Chat
from req  import ConversationRequest

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



S3_CLIENT = boto3.client(
    's3',
    region_name='ap-southeast-1',  # Specify your region
    aws_access_key_id='AKIAS6ZMFNZVXTOBUFH4',  # Replace with your access key ID
    aws_secret_access_key='3vw0HcrRb1MWTnTUFCymw/t/HcCl9OFgiMyUAvKW'  # Replace with your secret access key
)

S3_BUCKET_NAME="pdfchat-thesis"


def upload_to_s3(file: UploadFile, bucket: str) -> str:
    try:
        file_content = file.file.read()
        file_key = f"uploads/{file.filename}"
        S3_CLIENT.put_object(Bucket=bucket, Key=file_key, Body=file_content)
        return file_key
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise Exception(f"S3 upload error: {str(e)}")

def get_file_from_s3(file_key: str) -> BytesIO:
    try:
        response = S3_CLIENT.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        return BytesIO(response['Body'].read())
    except Exception as e:
        raise Exception(f"S3 retrieval error: {str(e)}")


class PresignedUrlRequest(BaseModel):
    filename: str
    content_type: str

class PresignedUrlResponse(BaseModel):
    url: str
    fields: Dict[str, str]

@app.post("/generate-presigned-url/")
async def generate_presigned_url(request: PresignedUrlRequest) -> PresignedUrlResponse:
    try:
        # Generate a pre-signed URL for uploading
        response = S3_CLIENT.generate_presigned_post(
            Bucket=S3_BUCKET_NAME,
            Key=f"uploads/{request.filename}",
            ExpiresIn=3600  # URL expiration time in seconds
        )

        return PresignedUrlResponse(
                   url=response['url'],
                   fields=response['fields']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating pre-signed URL: {str(e)}")


@app.post("/chat/")
async def new_chat(
    request: Request,
    db: Session = Depends(get_db),
    file_key: str = Query(..., description="The key of the file in S3 bucket")
):
    try:
        # Retrieve and process files from S3
        pdf_file_content = get_file_from_s3(file_key)

        # Assuming get_pdf_text can handle raw file content
        raw_text = get_pdf_text([pdf_file_content])
        text_chunks = get_text_chunks(raw_text)

        # Use the file_key to derive the filename if needed
        filename = file_key.split('/')[-1]

        # Add data to Chroma DB
        documents, ids, metadatas, embeddings = get_docs_to_add_vectorstore(text_chunks, filename)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # Create and save a new ChatHistory record
        chat_history = ChatHistory(title=filename, context=raw_text)
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
async def talk_with_gpt(question: ConversationRequest, db: Session = Depends(get_db)):
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
