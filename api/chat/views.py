import os
import json
import uuid
from io import BytesIO
from typing import List
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import JSONParser, MultiPartParser
from .models import ChatHistory, Chat, UploadedFile
from .serializers import ConversationRequestSerializer
from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document


# Initialize environment variables
google_api_key = os.getenv("GOOGLE_API_KEY",'AIzaSyBnuQcjTqmhKhXeLCaEYABsc8GyCflExVU' )

# Initialize Chroma DB client
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
    Provide a detailed answer to the question based on the context below. If the exact answer is not in the context, provide an answer that is closely related to the question based on the information available. Ensure that the answer is comprehensive and written in a long paragraph format, providing as much relevant information as possible. If necessary, make educated assumptions or connections, but ensure that they are relevant to the context provided.

    If the context contains images, interpret and describe them based on their content or labels. If a table of contents is identified, use it to structure the answer and provide the relevant sections that correspond to the question.

    Context (including text and images):\n {context}\n
    Question:\n {question}\n
    Answer:
"""


prompt = PromptTemplate.from_template(template)

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

def get_text_chunks_page_wise(pdf_docs: List[BytesIO]) -> List[str]:
    chunks = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                chunks.append(f"Page {page_num + 1}:\n{page_text.strip()}")
    return chunks

def get_limited_text_chunks(text: str, max_pages: int = 5) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks[:max_pages]  # Limit to a certain number of pages or chunks



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

@api_view(['POST'])
def upload_file(request):
    if 'file' not in request.FILES:
        return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

    file = request.FILES['file']

    # Read file content
    file_content = file.read()

    # Save to database
    try:
        uploaded_file = UploadedFile.objects.create(
            filename=file.name,
            content=file_content
        )
        return Response({"message": "File uploaded successfully", "id": uploaded_file.id, "file_name": uploaded_file.filename}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@parser_classes([MultiPartParser])
def new_chat(request):
    file_key = request.data.get('id')
    if not file_key:
        return Response({"error": "file_key is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Fetch the document content from the database using the file_key (assuming file_key is the ID)
        uploaded_file = UploadedFile.objects.get(id=file_key)

        # Extract the text from the PDF file content if not already saved
        if not uploaded_file.text_content:
            pdf_file_content = BytesIO(uploaded_file.content)  # Convert binary content to BytesIO
            raw_text = get_pdf_text([pdf_file_content])
            uploaded_file.text_content = raw_text
            uploaded_file.save()

        chat_history = ChatHistory(title=uploaded_file.filename, context=uploaded_file)
        chat_history.save()

        return JsonResponse(chat_history.to_dict())

    except UploadedFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_chats(request):
    try:
        chat_histories = ChatHistory.objects.all()
        data = [chat_history.to_dict_with_out_context() for chat_history in chat_histories]
        return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def talk_with_gpt(request):
    serializer = ConversationRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    chat_id = serializer.validated_data['chat_id']
    question = serializer.validated_data['question']

    try:
        chat_history = ChatHistory.objects.get(id=chat_id)
        
        # Use limited text chunks (e.g., process only the first 5 pages or chunks)
        documents = [Document(page_content=chunk) for chunk in get_limited_text_chunks(chat_history.context.text_content)]

        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": documents, "question": question}, return_only_outputs=True)
        
        answer = response.get("output_text", "No answer available.")
        chat = Chat(chat_history_id=chat_id, question=question, answer=answer)
        chat.save()

        return JsonResponse(chat.to_dict())
    
    except ChatHistory.DoesNotExist:
        return Response({"error": "ChatHistory not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['GET'])
def get_all_chats(request):
    chat_id = request.query_params.get('chat_id')
    if not chat_id:
        return Response({"error": "chat_id is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        chats = Chat.objects.filter(chat_history_id=chat_id)
        data = [chat.to_dict() for chat in chats]
        return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
