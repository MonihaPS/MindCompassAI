# rag_service.py 

import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage   # ← NEW
from sentence_transformers import SentenceTransformer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAGService:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file")

        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print("Initializing Groq LLM...")
        try:
            self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
            test = self.llm.invoke("Say hello").content
            print("Groq LLM test successful")
        except Exception as e:
            raise ValueError(f"Failed to initialize Groq LLM: {str(e)}")

        # Load knowledge base
        knowledge_base_path = "knowledge_base.json"
        if not os.path.exists(knowledge_base_path):
            raise FileNotFoundError(f"Knowledge base file not found: {knowledge_base_path}")

        print(f"Loading knowledge base from {knowledge_base_path}...")
        with open(knowledge_base_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        self.chunks = [doc['text'] for doc in self.documents]
        print(f"Loaded {len(self.chunks)} chunks.")

        # Normalize embeddings for cosine similarity
        print("Encoding & normalizing chunks...")
        embeddings = self.embedder.encode(self.chunks, show_progress_bar=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.dimension = embeddings.shape[1]
        print("Building FAISS IndexFlatIP (cosine similarity)...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

        print("RAG service initialized successfully.\n")

    def retrieve(self, query, k=5):
        if not query.strip():
            return []

        query_emb = self.embedder.encode([query])
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        distances, indices = self.index.search(query_emb.astype(np.float32), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and dist >= 0.35:          # cosine threshold
                results.append(self.chunks[idx])

        if not results:
            return ["No specific matching information found in knowledge base. Responding based on general mental health knowledge."]

        return results

    def generate_insight(self, emotion, confidence, additional_context=""):
        query = f"Mental health insights for {emotion} with confidence {confidence:.2f}. {additional_context}"
        retrieved = self.retrieve(query)
        context = "\n\n".join(retrieved)

        messages = [
            SystemMessage(content=(
                "You are a supportive AI companion for mental health awareness. "
                "Provide empathetic, non-diagnostic advice based on retrieved information. "
                "Always suggest professional help for serious issues. Do not diagnose."
            )),
            HumanMessage(content=(
                f"Detected emotion: {emotion}\n"
                f"Retrieved context:\n{context}\n\n"
                f"Generate a safe, helpful response about potential mental health implications."
            ))
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()

    def generate_chat_response(self, user_message, emotion_context=""):
        query = f"{user_message} in context of emotion {emotion_context}".strip()
        retrieved = self.retrieve(query)
        context = "\n\n".join(retrieved)

        messages = [
            SystemMessage(content=(
                "You are a kind, empathetic mental health awareness companion. "
                "Answer helpfully and realistically using the provided context when relevant. "
                "If little/no context matches, respond naturally and responsibly. "
                "Never diagnose. Always recommend professional help for serious concerns."
            )),
            HumanMessage(content=(
                f"User question/message: {user_message}\n"
                f"Current detected emotion context: {emotion_context}\n"
                f"Retrieved relevant information:\n{context}\n\n"
                f"Respond empathetically, supportively, and concisely."
            ))
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()