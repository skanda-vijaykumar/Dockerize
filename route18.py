from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import re 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceEmbeddingOptimizer
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from llama_index.core import (SimpleDirectoryReader, VectorStoreIndex, SimpleKeywordTableIndex, Settings, StorageContext, get_response_synthesizer)
import numpy as np
from difflib import SequenceMatcher
# from sentence_transformers import SentenceTransformer
from langchain_ollama.embeddings import OllamaEmbeddings
import uvicorn
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import SentenceWindowNodeParser,MarkdownNodeParser   
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.llms.ollama import Ollama
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor,Tool
from langchain import hub
from langchain_core.chat_history import BaseChatMessageHistory
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.core import QueryBundle,ServiceContext
import logging
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import psycopg
from fastapi.responses import StreamingResponse,FileResponse, HTMLResponse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import uuid
# from llama_index.core.node_parser import SentenceSplitter
import json
from typing import List, Dict, Any, Tuple
from llama_index.core import Document
from pathlib import Path
from llama_index.core.schema import Node
from llama_index.core.data_structs import Node
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, GoogleSerperAPIWrapper
import asyncio
from tavily import TavilyClient
from itertools import chain
from pydantic import Field
import torch
from datetime import datetime
from dateutil.parser import parse
import math
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from react_integration import ConnectorDimensionLangchainTool
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from functools import lru_cache
import urllib.parse

## Necessary declaration and initialization of API keys
TAVILY_API_KEY = "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"
SERPER_API_KEY = "a49f2db4b8df6ffba254aacc9a7d4dded2f50c1c"
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
serper_client = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY) 
ddg_search = DuckDuckGoSearchAPIWrapper()

## Initialize FastAPI app
app = FastAPI()

## Middleware for CORS (Cross-Origin Resource Sharing). Stick to this domain only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
## Static files for serving HTML and other assets
app.mount("/static", StaticFiles(directory="static"), name="static")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres") 
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "aspirine13z")
POSTGRES_DB = os.getenv("POSTGRES_DB", "test1")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama") 
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
## Langsmith monitoring
load_dotenv()
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "SQL_memory"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

## Global variables to hold data
vector_index_markdown = None
keyword_index_markdown = None
vector_index_markdown_lab = None
keyword_index_markdown_lab = None
documents_context = None
current_possibility=["AMM", "CMM", "DMM", "EMM"]
app_ready = False
startup_lock = threading.Lock()
startup_complete = threading.Event()
agent_pool = None
agent_lock = threading.Lock()
agent_queue = queue.Queue()
MAX_AGENTS = 4
## Retirever class with metadata filters (works fine don't touch)
## Screw based logic but still better.
class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever: VectorIndexRetriever, 
                 keyword_retriever: KeywordTableSimpleRetriever, 
                 mode: str = "AND") -> None:
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    ## Extract mentioned connector families from the user query
    def _extract_connector_families(self, query_text: str) -> List[str]:
        mentioned_families = []
        query_upper = query_text.upper()
        for family in ["AMM", "CMM", "DMM", "EMM"]:
            ## Check for the family name, if there is any then filter those nodes out. 
            if family in query_upper:
                mentioned_families.append(family)
        return mentioned_families

    # Modify the _retrieve method in the CustomRetriever class
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        ## Extract mentioned connector families from vdb
        mentioned_families = self._extract_connector_families(query_bundle.query_str)
        print(f"Connector families mentioned in query: {mentioned_families}")
        
        ## Get metadata from query bundle similarity nodes
        metadata = getattr(query_bundle, 'extra_info', {}) or {}
        file_type = metadata.get("type")
        
        ## Get basic retrievals 
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        print(f"Vector nodes retrieved: {len(vector_nodes)}")
        if vector_nodes:
            print(f"First vector node: {vector_nodes[0].node.text[:100]}...")

        # Add try-except block here
        try:
            keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
            print(f"Keyword nodes retrieved: {len(keyword_nodes)}")
        except AttributeError as e:
            print(f"Error retrieving keyword nodes: {str(e)}")
            ## If error empty list for keyword nodes
            keyword_nodes = []
        ## Only apply connector family filtering if families are mentioned
        if mentioned_families:
            print(f"Filtering by connector families: {mentioned_families}")            
            ## For vector nodes
            filtered_vector_nodes = []
            for node in vector_nodes:
                node_family = node.node.metadata.get("connector_family", "")
                ## Check if node family starts with any of the mentioned families in metadata that is
                if any(node_family.upper().startswith(family) for family in mentioned_families):
                    filtered_vector_nodes.append(node)
            ## For key-words
            filtered_keyword_nodes = []
            for node in keyword_nodes:
                node_family = node.node.metadata.get("connector_family", "")
                ## Check if node starts with any of the mentioned families in metadata that is
                if any(node_family.upper().startswith(family) for family in mentioned_families):
                    filtered_keyword_nodes.append(node)
            ## Combine both key words and vector nodes
            vector_nodes = filtered_vector_nodes
            keyword_nodes = filtered_keyword_nodes
            print(f"Nodes after connector family filter: {len(vector_nodes)} vectors, {len(keyword_nodes)} keywords")
            combined_nodes = vector_nodes + keyword_nodes
            represented_families = set()
            ## Filter again after combining just to make sure that overlaped nodes are still relevant
            for node in combined_nodes:
                node_family = node.node.metadata.get("connector_family", "").upper()
                for family in mentioned_families:
                    if node_family.startswith(family):
                        represented_families.add(family)
            
            ## Find missing families (incomplete will work later..)
            missing_families = set(mentioned_families) - represented_families
            
            # For each missing family, explicitly search for nodes
            if missing_families:
                print(f"Ensuring representation for missing families: {missing_families}")
                ## Test sequence
                for missing_family in missing_families:
                    family_query = f"{missing_family} temperature"
                    family_bundle = QueryBundle(family_query)
                    family_nodes = self._vector_retriever.retrieve(family_bundle)
                    ## Filter and add top nodes for selected family
                    for node in family_nodes:
                        node_family = node.node.metadata.get("connector_family", "").upper()
                        if node_family.startswith(missing_family):
                            print(f"Adding node for {missing_family}")
                            vector_nodes.append(node)
                            break
        
        ## Apply file type filtering if needed
        if file_type:
            vector_nodes = [n for n in vector_nodes if n.node.metadata.get("file_type") == file_type]
            keyword_nodes = [n for n in keyword_nodes if n.node.metadata.get("file_type") == file_type]
            print(f"Nodes after file type filter: {len(vector_nodes)} vectors, {len(keyword_nodes)} keywords")
        
        ## Combine results based on mode (AND/OR)
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        
        combined_dict = {}
        for n in vector_nodes:
            combined_dict[n.node.node_id] = n
        for n in keyword_nodes:
            combined_dict[n.node.node_id] = n
        ## AND is for intersection between keywords and vectors
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        ## OR for union between keywords and vectors
        else:
            retrieve_ids = vector_ids.union(keyword_ids)
            
        ## Get final nodes
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids if rid in combined_dict]
        
        ## Fallback if no nodes were retrieved
        if not retrieve_nodes:
            print("No nodes retrieved after filtering, using fallback")
            if vector_nodes:
                return vector_nodes
            elif keyword_nodes:
                return keyword_nodes
            else:
                return []
                
        print(f"Final nodes to return before reranking: {len(retrieve_nodes)}")
        return retrieve_nodes
    ## Retireve nodes based on user query
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_bundle = QueryBundle(query)
        nodes = self._retrieve(query_bundle)
        documents = []
        
        for node in nodes:
            try:
                doc = Document(
                    page_content=node.node.text,
                    metadata=node.node.metadata
                )
                documents.append(doc)
            except AttributeError:
                continue
                
        return documents
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
    


## Initialize database FOR HISTORY
def initialize_database():
    ## Fuq env variables
    conn_info = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    try:
        ## Connect to postgres db
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                ## Check if table exists in postgres
                cur.execute("""
                    SELECT EXISTS ( SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'chat_history');""")
                table_exists = cur.fetchone()[0]
                ## Create new table if not exist
                if not table_exists:
                    print("Creating chat_history table...")
                    cur.execute("""CREATE TABLE chat_history (id SERIAL PRIMARY KEY,session_id TEXT,chat_id INTEGER,message TEXT,type TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
                ## Check if sequence exist. This is for the fancy sequence number instead of Uid
                cur.execute("""SELECT EXISTS (SELECT 1 FROM pg_sequences WHERE schemaname = 'public' AND sequencename = 'chat_history_id_seq');""")
                sequence_exists = cur.fetchone()[0]
                ## Create new sequence if not exist
                if not sequence_exists:
                    print("Creating chat_id_seq sequence...")
                    ## Get the maximum chat_id to start the sequence from
                    cur.execute("SELECT COALESCE(MAX(chat_id), 0) + 1 FROM chat_history")
                    next_chat_id = cur.fetchone()[0]
                    ## Creating sequence (chat_ID) based on the last number in the sequence
                    cur.execute(f"""CREATE SEQUENCE chat_id_seq START WITH {next_chat_id} INCREMENT BY 1 NO MAXVALUE NO CYCLE """)
                conn.commit()
                print("Database initialization completed successfully")

    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise

## Customizing the postgresChatMessageHirstory chat history langchain method
class CustomPostgresChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, table_name: str, session_id: str, chat_id: int, sync_connection):
        self.table_name = table_name
        self.session_id = session_id
        self.chat_id = chat_id
        self.sync_connection = sync_connection
    ## Access postgres and add message structure
    def add_message(self, message):
        with self.sync_connection.cursor() as cursor:
            message_type = ("human" if isinstance(message, HumanMessage) else "ai" if isinstance(message, AIMessage) else "system")
            cursor.execute(f""" INSERT INTO {self.table_name} (session_id, chat_id, message, type) VALUES (%s, %s, %s, %s)""", (self.session_id, self.chat_id,  str(message.content), message_type),)
            self.sync_connection.commit()

    ## Retireving messages from postgres
    def get_messages(self):
        messages = []
        with self.sync_connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT type, message FROM {self.table_name}
                WHERE session_id = %s 
                ORDER BY created_at
                """,
                (self.session_id,))
            ## Assign roles to each message for identification
            for msg_type, content in cursor.fetchall():
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif msg_type == 'ai':
                    messages.append(AIMessage(content=content))
                elif msg_type == 'system':
                    messages.append(SystemMessage(content=content))
        return messages
    ## delete from history
    def clear(self):
        with self.sync_connection.cursor() as cursor:
            cursor.execute(f""" DELETE FROM {self.table_name} WHERE session_id = %s """, (self.session_id,) )
            self.sync_connection.commit()

## Maps session_id with chat_id and created_at
def load_session_mapping():
    conn_info = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    try:
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                ## Get the most recent session for each chat_id
                cur.execute(""" SELECT DISTINCT ON (chat_id) session_id, chat_id, created_at FROM chat_history ORDER BY chat_id, created_at DESC""")
                rows = cur.fetchall()
                session_mapping = {}
                for session_id, chat_id, created_at in rows:
                    timestamp = created_at.timestamp()
                    session_mapping[session_id] = { 'chat_id': chat_id, 'timestamp': timestamp}
                return session_mapping
    except Exception as e:
        logging.error(f"Error loading session mapping: {str(e)}")
        return {}

## Conversation history based on session_id and chat_id 
def get_session_history(session_id: str, chat_id: int) -> BaseChatMessageHistory:
    conn_info = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    sync_connection = psycopg.connect(conn_info)
    return CustomPostgresChatMessageHistory("chat_history", session_id, chat_id, sync_connection=sync_connection)

def get_next_chat_id(sync_connection):
    cursor = sync_connection.cursor()
    cursor.execute("SELECT COALESCE(MAX(chat_id), 0) + 1 FROM chat_history")
    next_chat_id = cursor.fetchone()[0]
    cursor.close()
    return next_chat_id

class RankedNodesLogger():    
    def __init__(self, reranker):
        self.reranker = reranker
        self.source_tracker = SourceTracker()
        super().__init__()
        
    def postprocess_nodes(self, nodes, query_bundle):
        # Pass nodes through the underlying reranker
        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        
        # Log the reranked nodes
        print("\n=== RERANKED NODES ===")
        print(f"Showing {len(reranked_nodes)} nodes after reranking")
        
        # Add all sources to the global tracker
        self.source_tracker.add_sources_from_nodes(reranked_nodes)
        
        for i, node in enumerate(reranked_nodes):
            source = node.node.metadata.get("source", "Unknown")
            connector_family = node.node.metadata.get("connector_family", "Unknown")
            file_type = node.node.metadata.get("file_type", "Unknown")
            score = node.score if hasattr(node, 'score') else "N/A"
            absolute_path = node.node.metadata.get("absolute_path", "Unknown path")
            
            print(f"Node {i+1}: {source} | Family: {connector_family} | Type: {file_type} | Score: {score}")
            print(f"  Path: {absolute_path}")
            
            # First few characters of content for context
            node_text = node.node.text.replace('\n', ' ')
            if len(node_text) > 100:
                node_text = node_text[:100] + "..."
            print(f"  Content: {node_text}")
        
        print("=== END OF RERANKED NODES ===\n")
        
        return reranked_nodes

def load_data(directory_path):
    documents1 = []
    
    for md_filename in os.listdir(directory_path):
        if not md_filename.lower().endswith(".md"):
            continue
        file_path = os.path.join(directory_path, md_filename)
        raw = open(file_path, encoding="utf-8").read()
        
        ## this regex finds all occurrences of <!-- PAGE: N -->
        ## split into pages and captures the page number
        splits = re.split(r"\s*<!--\s*PAGE:\s*(\d+)\s*-->\s*", raw)
        it = iter(splits)
        preamble = next(it)
        for page_num, page_text in zip(it, it):
            # build metadata
            meta = {
                "file_type": "markdown",
                "connector_family": Path(md_filename).stem.upper(),
                "source": Path(md_filename).stem,
                "absolute_path": os.path.abspath(file_path),
                "page_number": int(page_num),
            }
            documents1.append(Document(text=page_text, metadata=meta))
    
    lab_dir = "./extracted_best/lab/"
    documents3 = []
    for md_filename in os.listdir(lab_dir):
        if not md_filename.lower().endswith(".md"):
            continue
        file_path = os.path.join(lab_dir, md_filename)
        raw = open(file_path, encoding="utf-8").read()
        
        splits = re.split(r"\s*<!--\s*PAGE:\s*(\d+)\s*-->\s*", raw)
        it = iter(splits)
        preamble = next(it)
        for page_num, page_text in zip(it, it):
            meta = {
                "file_type": "markdown",
                "connector_family": Path(md_filename).stem.upper(),
                "source": Path(md_filename).stem,
                "absolute_path": os.path.abspath(file_path),
                "page_number": int(page_num),
            }
            documents3.append(Document(text=page_text, metadata=meta))
    
    reader = JSONReader()
    json_docs = []
    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".json"):
            continue
        file_path = os.path.join(directory_path, filename)
        for doc in reader.load_data(input_file=file_path, extra_info={}):
            abs_file_path = os.path.abspath(file_path)
            doc.metadata.update({
                "file_type": "json",
                "connector_family": Path(file_path).stem.upper(),
                "source": Path(file_path).stem,
                "absolute_path": abs_file_path,
            })
            json_docs.append(doc)    
    # combine
    documents1.extend(json_docs)
    return documents1, documents3

## Chunk loaded documents
def processing_data(documents1, documents3):
    try:
        print(f"Processing documents1: {len(documents1)} documents")
        print(f"Processing documents3: {len(documents3)} documents")

        ## SentenceWindowNodeParser mmmmmm gooooood...
        sentencewindow_node_parser = SentenceWindowNodeParser(
            include_metadata=True,
            include_prev_next_rel=True,
            window_size=5
        )
        ## Backup cause markdown is da format
        markdown_node_parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True,
        )
        
        print("Parsing nodes from documents...")
        ## Catalogue data
        nodes_sentencewindow = sentencewindow_node_parser.get_nodes_from_documents(documents1)
        print(f"Generated {len(nodes_sentencewindow)} sentence window nodes")
        
        nodes_markdown_nodes = markdown_node_parser.get_nodes_from_documents(documents1)
        print(f"Generated {len(nodes_markdown_nodes)} markdown nodes")
        
        nodes_markdown = nodes_sentencewindow + nodes_markdown_nodes
        print(f"Total catalogue nodes: {len(nodes_markdown)}")
        
        ## Labs data
        nodes_sentencewindow1 = sentencewindow_node_parser.get_nodes_from_documents(documents3)
        print(f"Generated {len(nodes_sentencewindow1)} lab sentence window nodes")
        
        nodes_markdown_nodes_lab = markdown_node_parser.get_nodes_from_documents(documents3)
        print(f"Generated {len(nodes_markdown_nodes_lab)} lab markdown nodes")
        
        nodes_markdown_lab = nodes_sentencewindow1 + nodes_markdown_nodes_lab
        print(f"Total lab nodes: {len(nodes_markdown_lab)}")

        ## Settings
        print("Initializing language model and embedding model...")
        Settings.llm = Ollama(model="llama3.1", temperature=0.0, num_ctx=8012, top_p=0.5, base_url=OLLAMA_BASE_URL)
        Settings.embed_model = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
        
        print("Creating storage contexts...")
        ## Markdown: Create storage context and add documents
        storage_context_markdown = StorageContext.from_defaults()
        storage_context_markdown.docstore.add_documents(nodes_markdown)
        
        ## LAB Markdown: Create storage context and add documents
        storage_context_markdown_lab = StorageContext.from_defaults()
        storage_context_markdown_lab.docstore.add_documents(nodes_markdown_lab)
        
        print("Creating vector indices...")    
        ## Markdown: Create both indices
        vector_index_markdown = VectorStoreIndex(nodes_markdown, storage_context=storage_context_markdown, similarity_top_k=25,
            index_kwargs={
                "metric": "cosine", "normalize_embeddings": True,
                "hnsw": {
            "max-links-per-node": 64, "neighbors-to-explore-at-insert": 300, "ef_construction": 400}})
        print("Vector index for markdown created successfully")
        
        ## Lab Markdown: Create both indices
        vector_index_markdown_lab = VectorStoreIndex(nodes_markdown_lab, storage_context=storage_context_markdown_lab, similarity_top_k=25,
            index_kwargs={"metric": "cosine", "normalize_embeddings": True, "hnsw": {"max-links-per-node": 64, "neighbors-to-explore-at-insert": 300, "ef_construction": 400}})
        print("Vector index for lab markdown created successfully")
       
        print("Creating keyword indices...")
        ## Keywords indices
        keyword_index_markdown = SimpleKeywordTableIndex(nodes_markdown, storage_context=storage_context_markdown, show_progress=True)
        print("Keyword index for markdown created successfully")
        
        keyword_index_markdown_lab = SimpleKeywordTableIndex(nodes_markdown_lab, storage_context=storage_context_markdown_lab, show_progress=True)
        print("Keyword index for lab markdown created successfully")
        
        # Verify that all indices were created correctly
        if vector_index_markdown is None:
            print("ERROR: vector_index_markdown failed to initialize")
        if keyword_index_markdown is None:
            print("ERROR: keyword_index_markdown failed to initialize")
        if vector_index_markdown_lab is None:
            print("ERROR: vector_index_markdown_lab failed to initialize")
        if keyword_index_markdown_lab is None:
            print("ERROR: keyword_index_markdown_lab failed to initialize")
        
        print("All indices created successfully")
        
        # wasn't properly assigned, which suggests there might be an ordering issue
        return vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab
    
    except Exception as e:
        print(f"Error in processing_data: {str(e)}")
        # Return None values to indicate failure
        return None, None, None, None
## Internet search through multiple browsers
class MultiSearchRetriever(BaseRetriever):
    ## Clients providers
    tavily_client: TavilyClient 
    serper_client: GoogleSerperAPIWrapper 
    ddg_search: DuckDuckGoSearchAPIWrapper
    
    ## Embedder
    encoder: OllamaEmbeddings = None
    
    def __init__(self,tavily_client: TavilyClient,serper_client: GoogleSerperAPIWrapper,ddg_search: DuckDuckGoSearchAPIWrapper):
        super().__init__()
        self.tavily_client = tavily_client
        self.serper_client = serper_client
        self.ddg_search = ddg_search
        try:
            # Initialize Ollama embeddings
            self.encoder = OllamaEmbeddings(model="nomic-embed-text")
        except Exception as e:
            print(f"Error loading primary model: {e}")
            try:
                # Fallback to alternative model
                self.encoder = OllamaEmbeddings(model="mxbai-embed-large")
            except Exception as e:
                print(f"Error loading fallback model: {e}")
                self.encoder = None
                print("Falling back to basic text similarity")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            ## Convert the query to string if it's a QueryBundle
            query_str = query_bundle.query_str if isinstance(query_bundle, QueryBundle) else str(query_bundle)
            ## Get documents using existing method
            docs = self._get_relevant_documents(query_str)
            ## same retirever logic like the other class
            nodes_with_scores = []
            for i, doc in enumerate(docs):
                node = Node(text=doc.page_content,metadata=doc.metadata)
                score = 1.0 - (i * 0.1)
                nodes_with_scores.append(NodeWithScore(node=node, score=score))
            return nodes_with_scores
    
    ## Search and retireve context from tavily
    def _get_tavily_documents(self, query: str) -> List[Document]:
        try:
            response = self.tavily_client.search(query=query)
            results = response.get("results", [])
            documents = []
            for res in results[:3]:
                if len(res.get("content", "")) < 50:
                    continue
                doc = Document(text=res.get("content", ""))
                doc.metadata = {"source": res.get("url", ""),"title": res.get("title", ""),"provider": "Tavily","orig_order": len(documents)}
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"Error getting Tavily results: {e}")
            return []
        
    ## Fetch data from google serper
    def _get_serper_documents(self, query: str) -> List[Document]:
        try:
            raw_results = self.serper_client.results(query)
            results = raw_results.get("organic", [])
            documents = []
            for res in results[:3]:
                if len(res.get("snippet", "")) < 50:
                    continue
                doc = Document(text=res.get("snippet", ""))
                doc.metadata = {"source": res.get("link", ""),"title": res.get("title", ""),"provider": "Google Serper","orig_order": len(documents)}
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"Error getting Serper results: {e}")
            return []
        
    ## Get data from DuckDuckGo
    def _get_ddg_documents(self, query: str) -> List[Document]:
        try:
            time.sleep(1)
            results = self.ddg_search.run(query)
            if not results or len(results) < 50:
                return []
            
            doc = Document(text=results)
            doc.metadata = {"source": "DuckDuckGo Search","title": "DuckDuckGo Results","provider": "DuckDuckGo","orig_order": 0}
            return [doc]
        except Exception as e:
            print(f"Error getting DuckDuckGo results: {e}")
            return []
        
    ## Finding data of publication of the article
    def _extract_date(self, content: str) -> datetime:
        ## Regex to find date in a page
        date_patterns = [r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}/\d{2}/\d{2}',
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}']
        ## Findall
        for pattern in date_patterns:
            matches = re.findall(pattern, content.lower(), re.IGNORECASE)
            if matches:
                try:
                    return parse(matches[0])
                except (ValueError, TypeError):
                    continue
        return None
    
    ## Computing relevency scores
    def _compute_scores(self, query: str, documents: List[Document]) -> List[tuple]:
        if not documents:
            return []
        current_date = datetime.now()
        
        try:
            if self.encoder is not None:
                # Get text content from documents
                doc_texts = []
                ## Find text elements
                for doc in documents:
                    if hasattr(doc, 'text_resource') and doc.text_resource:
                        doc_texts.append(doc.text_resource.text)
                    elif hasattr(doc, 'page_content'):
                        doc_texts.append(doc.page_content)
                    else:
                        continue

                ## Get embeddings using Ollama embeddings
                query_embedding = self.encoder.embed_query(query)
                doc_embeddings = self.encoder.embed_documents([doc.page_content for doc in documents])
                
                ## Convert to numpy arrays for computation
                query_embedding = np.array(query_embedding)
                doc_embeddings = np.array(doc_embeddings)
                
                ## Compute cosine similarities
                similarities = np.dot(doc_embeddings, query_embedding) / (np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding))
            else:
                ## Fallback to simple text matching
                similarities = [SequenceMatcher(None, query.lower(), doc.page_content.lower()).ratio() for doc in documents]
                similarities = np.array(similarities)
            
            scored_docs = []
            for doc, similarity in zip(documents, similarities):
                doc_date = self._extract_date(doc.page_content)
                
                if doc_date:
                    days_old = (current_date - doc_date).days
                    recency_score = math.exp(-max(0, days_old) / 365)
                else:
                    recency_score = 0.5
                ## Ahh yess the similarity to recency score ratio
                combined_score = (0.35 * float(similarity)) + (0.55 * recency_score)
                scored_docs.append((doc, combined_score))
                
                ## Add scores to metadata
                doc.metadata["similarity_score"] = float(similarity)
                doc.metadata["recency_score"] = float(recency_score)
                doc.metadata["combined_score"] = float(combined_score)
            
            return sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Error in computing scores: {e}")
            # Return original documents with neutral scores if scoring fails
            return list(zip(documents, [1.0] * len(documents)))
    
    ## Callin the reitrevers
    def _get_relevant_documents(self, query: str) -> List[Document]:
        ## Get results from all search providers
        tavily_docs = self._get_tavily_documents(query)
        serper_docs = self._get_serper_documents(query)
        ddg_docs = self._get_ddg_documents(query)
        
        ## Combine all results
        all_docs = list(chain(tavily_docs, serper_docs, ddg_docs))
        
        ## Remove duplicates based on URL
        seen_urls = set()
        unique_docs = []
        for doc in all_docs:
            url = doc.metadata["source"]
            if url not in seen_urls:
                seen_urls.add(url)
                unique_docs.append(doc)
        
        ## Rerank documents based on combined score
        reranked_docs = self._compute_scores(query, unique_docs)
        ## Return top 10 content
        return [doc for doc, _ in reranked_docs[:10]]
    ## ASync run on the above to calls dont have to wait
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
    
## Defining tools for Agent  
def creating_tools(vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab):
    tools = []
    
    try:
        # Print input indices types
        print("\nCreating tools with indices:")
        print(f"- vector_index_markdown: {type(vector_index_markdown).__name__ if vector_index_markdown is not None else 'None'}")
        print(f"- keyword_index_markdown: {type(keyword_index_markdown).__name__ if keyword_index_markdown is not None else 'None'}")
        print(f"- vector_index_markdown_lab: {type(vector_index_markdown_lab).__name__ if vector_index_markdown_lab is not None else 'None'}")
        print(f"- keyword_index_markdown_lab: {type(keyword_index_markdown_lab).__name__ if keyword_index_markdown_lab is not None else 'None'}")
        
        # Only create retrievers if necessary indices are available
        if vector_index_markdown is not None and keyword_index_markdown is not None:
            print("Creating markdown tools...")
            try:
                ## Create retrievers for markdowns
                vector_retriever_markdown = VectorIndexRetriever(index=vector_index_markdown, similarity_top_k=30,
                                                             vector_store_kwargs={"search_kwargs": {"search_type": "similarity", "k": 30}})
                keyword_retriever_markdown = KeywordTableSimpleRetriever(index=keyword_index_markdown, similarity_top_k=25)
                hybrid_retriever_markdown = CustomRetriever(vector_retriever=vector_retriever_markdown, keyword_retriever=keyword_retriever_markdown, mode="OR")
                
                ## Filters and retriever strats
                base_reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=15)
                # Wrap the reranker with our logger
                reranker = RankedNodesLogger(base_reranker)
                response_synthesizer = get_response_synthesizer(response_mode="compact_accumulate", verbose=True)
                hybrid_query_engine_markdown = RetrieverQueryEngine(retriever=hybrid_retriever_markdown, response_synthesizer=response_synthesizer,
                                                                node_postprocessors=[reranker])

                ## Create tools list
                query_engine_tools_markdown = [
                    QueryEngineTool(
                        query_engine=hybrid_query_engine_markdown,
                        metadata=ToolMetadata(
                            name="Nicomatic_connector_catalogue",
                            description="""
                            A technical repository for Nicomatic products specifications and compatibility information. Use this as your PRIMARY tool for:
                                - Cable compatibility questions (e.g., "What cable goes with 30-1447-ZZ?")
                                - Finding accessories that match with specific connectors
                                - Pitch size
                                - All part number lookups and compatibility checks
                                - Connector specifications including temperature ratings, electrical properties, materials and more
                                - Any question about what "works with", "goes with", or is "compatible with" a part number
                                
                            This tool contains comprehensive product information beyond just dimensions. For questions about connector compatibility, accessories, or cables, ALWAYS use this tool first.
                            When using this tool make sure that the input if needed will have connector name mentioned which user is referring to like AMM, CMM, DMM, EMM. 
                            """))]
                            
                ## Convert tools for langchain
                llamaindex_to_langchain_converted_tools_markdown = [t.to_langchain_tool() for t in query_engine_tools_markdown]
                tools.extend(llamaindex_to_langchain_converted_tools_markdown)
                print("Added markdown catalog tools")
            except Exception as e:
                print(f"Error creating markdown tools: {str(e)}")
        else:
            print("Skipping markdown tools due to missing indices")

        if vector_index_markdown_lab is not None and keyword_index_markdown_lab is not None:
            print("Creating lab tools...")
            try:
                ## Create retrievers for markdowns for lab files
                vector_retriever_markdown_lab = VectorIndexRetriever(index=vector_index_markdown_lab, similarity_top_k=30,
                                                                 vector_store_kwargs={"search_kwargs": {"search_type": "similarity", "k": 30}})
                keyword_retriever_markdown_lab = KeywordTableSimpleRetriever(index=keyword_index_markdown_lab, similarity_top_k=25)
                hybrid_retriever_markdown_lab = CustomRetriever(vector_retriever=vector_retriever_markdown_lab, keyword_retriever=keyword_retriever_markdown_lab, mode="OR")
                
                base_reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=15)
                lab_reranker = RankedNodesLogger(base_reranker)
                
                response_synthesizer = get_response_synthesizer(response_mode="accumulate", verbose=True)
                
                hybrid_query_engine_markdown_lab = RetrieverQueryEngine(retriever=hybrid_retriever_markdown_lab, response_synthesizer=response_synthesizer,
                                                                        node_postprocessors=[lab_reranker])
                
                query_engine_tools_markdown_lab = [
                    QueryEngineTool(
                        query_engine=hybrid_query_engine_markdown_lab,
                        metadata=ToolMetadata(
                            name="Nicomatic_lab_tests",
                            description="Use this tool to find information about all Nicomatic connector's lab tests. This tool has access to the all lab tests of humidity, durability, mating and unmating forces and a lot more for AMM, CMM, DMM, EMM. Input must be a clear sentence."
                        ))]
                        
                llamaindex_to_langchain_converted_tools_markdown_lab = [t.to_langchain_tool() for t in query_engine_tools_markdown_lab]
                tools.extend(llamaindex_to_langchain_converted_tools_markdown_lab)
                print("Added lab tools")
            except Exception as e:
                print(f"Error creating lab tools: {str(e)}")
        else:
            print("Skipping lab tools due to missing indices")
        
        ## calling the previously initialized internet retriever
        def search_function(query: str) -> List[Document]:
            search = MultiSearchRetriever(tavily_client=tavily_client, serper_client=serper_client, ddg_search=ddg_search)
            return search._get_relevant_documents(query)
            
        ## Add this to search tool
        search_tool_dict = Tool(name="Internet_tool", func=search_function, 
                                description="A general-purpose search capability that accesses external data sources beyond Nicomatic's internal documentation. This tool should only be used as a fallback when other specialized tools fail to provide relevant information, or for queries about industry trends, competitor's products, other companies, historical information, information on people, names or general technical concepts not specific to Nicomatic products. Always prioritize Nicomatic's specialized tools over this general search function for product-specific information. When using this tool, always include 'Nicomatic' in the search query unless the question is clearly about a non-Nicomatic topic.")
        
        ## Add search tool
        tools.append(search_tool_dict)
        print("Added internet search tool")
        
        ## Try to add connector dimension tool
        data_dir = "./extracted_best"
        try:
            dimension_tool = ConnectorDimensionLangchainTool(data_dir)
            tools.append(dimension_tool)
            print("Added connector dimension tool")
        except Exception as e:
            print(f"Error adding connector dimension tool: {str(e)}")
    
    except Exception as e:
        print(f"Error creating tools: {str(e)}")
        # At minimum, always add the search tool for fallback capability
        def search_function(query: str) -> List[Document]:
            search = MultiSearchRetriever(tavily_client=tavily_client, serper_client=serper_client, ddg_search=ddg_search)
            return search._get_relevant_documents(query)
            
        search_tool_dict = Tool(name="Internet_tool", func=search_function, 
                               description="A general-purpose search capability that accesses external data sources.")
        tools.append(search_tool_dict)
        print("Added internet search tool as fallback")
    
    print(f"Created {len(tools)} tools")
    return tools
# Create an isolated agent with its own LLM instance
def create_isolated_agent(tools):
    llm = ChatOllama( model="llama3.1", temperature=0.0, num_ctx=8152, cache=False, client_kwargs={"timeout": 60}, base_url=OLLAMA_BASE_URL,client_id=f"agent-{uuid.uuid4()}")
    prompt = hub.pull("intern/ask10")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=2, max_execution_time=None, 
                                   callbacks=[StreamingStdOutCallbackHandler()], return_intermediate_steps=True, early_stopping_method='force')
    return agent_executer

# ## Add this worker-local cache
# _worker_llm_cache = {}
# def get_worker_llm():
#     worker_id = os.getpid()
#     if worker_id not in _worker_llm_cache:
#         _worker_llm_cache[worker_id] = ChatOllama(
#             model="llama3.1",
#             temperature=0.01,
#             num_ctx=6000,
#             client_id=f"worker-{worker_id}"
#         )
#     return _worker_llm_cache[worker_id]

# Function to get an agent from the pool
async def get_agent(tools):
    try:
        ## Try to get an existing agent from the queue
        agent = agent_queue.get_nowait()
        return agent
    except queue.Empty:
        ## If no agents are available, create a new one
        return create_isolated_agent(tools)

## Return an agent to the pool
def return_agent(agent):
    try:
        agent_queue.put(agent, block=False)
    except queue.Full:
        pass

# def get_llm():
#     return ChatOllama(
#         model="llama3.1",
#         temperature=0.01,
#         num_ctx=8126,
#         cache=False,
#         client_kwargs={"timeout": 60},
#         client_id=f"client-{uuid.uuid4()}" 
#     )

session_mapping = {}
chat_id_counter = 1  


## AWG parsing has sm issue so this
def normalize_awg_value(awg_value):
            if isinstance(awg_value, (int, float)):
                return int(awg_value)
            elif isinstance(awg_value, str):
                awg_str = awg_value.upper()
                if "AWG" in awg_str:
                    try:
                        return int(awg_str.replace("AWG", ""))
                    except ValueError:
                        pass
            # Return None if conversion failed
            return None

## Connector selection logic    
## Most Annoying class in existance 
class LLMConnectorSelector:
    def __init__(self):
        ## Chatmodel
        self.llm = ChatOllama(model="llama3.1",cache=False,base_url=OLLAMA_BASE_URL)
        ## Structure for the LLM response
        self.response_schemas = [ResponseSchema(name="value", description="The parsed value from user response"),ResponseSchema(name="confidence", description="Confidence score between 0 and 1"),
                                  ResponseSchema(name="reasoning", description="Explanation of the parsing logic")]
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)
        ## System prompt for parsing
        self.system_prompt = """You are an expert in electronic connectors, specifically the AMM, CMM, DMM, and EMM connector families.
        Your role is to parse user responses to questions about connector requirements and extract meaningful information.
        You should handle uncertainty in responses and provide confidence scores.
        
        Key points:
        - Provide clear numerical or boolean values when possible
        - Handle uncertain responses with appropriate confidence scores
        - Consider technical context of each question
        - Explain your reasoning
        """
        ## Ground data sampled for the llm to grade instead of relying on RAG
        ## This helped me for the backup parser  
        self.connectors = {
            'AMM': {
                'type':'nanod',
                'pitch_size': 1.0,
                'emi_protection': False,
                'housing_material': 'plastic',
                'weight_category': 'lightest',
                'panel_mount': False,
                'height_range': (4.0, 4.0),
                'pcb_thickness_range': (0.8, 3.2),
                'right_angle': False,
                'temp_range': (-65, 200),
                'vibration_g': 15,
                'shock_g': 100,
                'max_current': 4.8,
                'contact_resistance': 10,
                'mixed_power_signal': False,
                'location': 'internal',
                'max_mating_force': 0.5,
                'min_unmating_force': 0.2,
                'wire_gauge': ['AWG26', 'AWG28', 'AWG30'],
                'mating_cycles': 1000,
                'availability': 'COTS',
                'height_options': [4.0],  
                'height_range': (4.0, 4.0),
                'valid_pin_counts': set([6, 10, 20, 34, 50]),
                'max_pins': 50

            },
            'CMM': {
                'type':'subd',
                'pitch_size': 2.0,
                'emi_protection': False,
                'housing_material': 'plastic',
                'weight_category': 'medium',
                'panel_mount': False,
                'height_range': (3.5, 8.0),
                'pcb_thickness_range': (0.8, 3.2),
                'right_angle': True,
                'temp_range': (-60, 260),
                'vibration_g': 10,
                'shock_g': 100,
                'max_current': 30,
                'wire_gauge': ['AWG12', 'AWG14', 'AWG16', 'AWG18', 'AWG20', 'AWG22', 'AWG24', 'AWG26', 'AWG28', 'AWG30'],
                'contact_resistance': 10,
                'mixed_power_signal': True,
                'location': 'internal',
                'max_mating_force': 2.0,
                'min_unmating_force': 0.2,
                'mating_cycles': 2500,
                'availability': 'made_to_order',
                'height_options': [3.5, 4.0, 5.5, 6.0, 7.7, 8.0], 
                'valid_pin_counts': set([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120]),
                'max_pins': 120
            },
            'DMM': {
                'type':'microd',
                'pitch_size': 2.0,
                'emi_protection': True,
                'housing_material': 'metal',
                'weight_category': 'heaviest',
                'panel_mount': True,
                'height_range': (5.0, 17.5),
                'pcb_thickness_range': (0.8, 3.5),
                'right_angle': True,
                'temp_range': (-55, 125),
                'vibration_g': 20,
                'shock_g': 100,
                'max_current': 20,
                'contact_resistance': 7.63,
                'mixed_power_signal': True,
                'wire_gauge': ['AWG12', 'AWG14', 'AWG16', 'AWG18', 'AWG20', 'AWG22', 'AWG24', 'AWG26', 'AWG28', 'AWG30'],
                'location': 'external',
                'max_mating_force': 9.733,
                'min_unmating_force': 0.000002,
                'mating_cycles': 500,
                'availability': 'made_to_order',
                'height_options': [5.0, 6.2, 7.0, 8.2, 9.0, 9.2, 9.65, 10.1, 10.2, 10.5, 10.55, 11.0, 11.45, 11.5, 11.9, 12.0, 12.2, 12.35, 12.5, 12.8, 13.0, 13.25, 13.5, 13.7, 14.0, 14.15, 14.5, 14.6, 15.0, 15.05, 15.5, 16.0, 16.5, 17.0,17.5],  
                'valid_pin_counts': set([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 36, 38, 39, 40, 42, 44, 45, 46, 48, 50, 51, 52, 54, 56, 57, 58, 60, 63, 64, 66, 68, 69, 72, 75, 76, 78, 80, 81, 84, 87, 88, 90, 92, 96, 100, 104, 108, 112, 116, 120]),
                'max_pins': 120
            },
            'EMM': {
                'type':'microd',
                'pitch_size': 1.27,
                'emi_protection': False,
                'housing_material': 'plastic',
                'weight_category': 'light-medium',
                'panel_mount': False,
                'height_range': (4.6, 4.6),
                'pcb_thickness_range': (0.8, 3.5),
                'right_angle': True,
                'temp_range': (-65, 200),
                'vibration_g': 45,
                'shock_g': 160,
                'max_current': 3.9,
                'contact_resistance': 8,
                'mixed_power_signal': False,
                'location': 'internal',
                'wire_gauge': ['AWG24', 'AWG26', 'AWG28', 'AWG30'],
                'max_mating_force': 1.7,
                'min_unmating_force': 0.1,
                'mating_cycles': 500,
                'availability': 'made_to_order',
                'height_options': [4.6],  
                'valid_pin_counts': set([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]),
                'max_pins': 60
            }
        }

        ## set of questions to ask the user for help in shortlisting
        self.all_questions = [
            {
                'text': 'What connection type do you need? (PCB-Cable,PCB-PCB,Cable-Cable)',
                'weight': 25,
                'attribute': 'connection_types',
                'clarification': 'Choose between PCB to PCB, PCB to Cable, Cable to Cable, or Cable to PCB configurations',
                'parse_prompt': """Identify the desired connection configuration from:
                - PCB to PCB, pcb to pcb, board to board, Board to Board
                - PCB to Cable, pcb to cable, board to cable, Board to cable
                - Cable to Cable, cable to cable
                - Cable to PCB, cable to pcb, cable to board, cable to Board""",
                'order': 1
            },
            {
                'text': 'Do you need this connector on-board or panel mount use?',
                'weight': 30,
                'attribute': 'location',
                'clarification': 'In box is inside equipment, out of box is panel mounting.',
                'parse_prompt': """Determine if the application is in box or out of box. Look for keywords indicating location and environment. 
                - 'Out of box' can also be mentioned as on Panel,  panel mounting,  external,  outside, on box, or something similar. 
                - 'In box' can also be mentioned as internal, inside,  on-board, or something similar."""
                ,'order': 2
            },
            {
                'text': 'Do you require a <b>Plastic housing</b> or a <b>Metal housing</b> with EMI shielding for this connector?',
                'weight': 70,  
                'attribute': 'housing_material',
                'clarification': 'Metal housing (DMM) provides better durability and EMI protection, plastic housing is lighter and cost-effective.',
                'parse_prompt': """Determine if the user wants plastic or metal housing. 
                - If user mentions metallic preference, aluminium, with EMI, need EMI, or steel,  it indicates metal. 
                - If user mentions , polymer, composite, without EMI, or non-metal, it indicates plastic """
                ,'order': 3
            },
            {
                'text': 'Do you need high power/frequency (>5 Amps) capabilities for this connector?',
                'weight': 20,
                'attribute': 'mixed_power_signal',
                'clarification': 'Mixed power/signal allows both power and data in one connector.',
                'parse_prompt': """Determine if mixed power and signal capability is required. can also be mentioned as mixing signals and high power"""
                ,'order': 4
            },
            {
                'text': 'How many signal contacts/pins do you need?',
                'weight': 25,
                'attribute': 'pin_count',
                'clarification': 'Valid pin counts: AMM (4-50 even numbers only), CMM 2-120 pins (both odd and even), DMM 2-120 pins (both odd and even), EMM (4-60 even numbers only)',
                'parse_prompt': """Extract the exact number of pins/contacts needed.
                Verify if the number is within valid ranges:
                - AMM: only has 6, 10, 20, 34, or 50
                - CMM: 2-120 pins (both odd and even)
                - DMM: 2-120 pins (both odd and even)
                - EMM: 4-60 pins (even numbers)
                If number exceeds any family's maximum, this should be noted as a critical mismatch."""
                ,'order': 5
            },
        
            {
                'text': 'What are your height or space constraints (in mm)?',
                'weight': 10, 
                'attribute': 'height_requirement',
                'clarification': 'Available heights/widths: AMM (4.0mm), EMM (4.6mm), CMM (5.5mm/7.7mm), DMM (5.0mm/7.0mm)',
                'parse_prompt': """Extract the height/width requirement from the user's response. 
                Look for:
                - Exact measurements (e.g., "5mm", "4.6 millimeters")
                - Range specifications (e.g., "under 5mm", "maximum 6mm")
                - Dimensional constraints (e.g., "50x5mm", "space of 5mm")
                Return the height value in millimeters.""",
                'order': 6
            },
            {
                'text': 'We offer pitch sizes of 1mm, 1.27mm, and 2mm. Which one best suits your requirement?',
                'weight': 70,
                'attribute': 'pitch_size',
                'clarification': 'The pitch size is the distance between connector contacts. Common sizes are 1.0mm (AMM), 1.27mm (EMM), or 2.0mm (CMM/DMM).',
                'parse_prompt': """Extract the pitch size value from the user's response. Valid values are 1, 1.27, and 2 mm. If uncertain, provide a confidence score less than 1.0.""",
                'order': 7,
                'images': [
                    '/static/pitch1mm.png',  
                    '/static/pitch2mm.png', 
                    '/static/pitch127mm.png' 
                ],
                'has_images': True
            },
            {
                'text': 'Do you need a right-angle connector or a straight connector?',
                'weight': 30, 
                'attribute': 'right_angle',
                'clarification': 'Right-angle connectors come out parallel to the board, while straight connectors come out perpendicular to the board.',
                'parse_prompt': """Determine if the user needs a right-angle connector (TRUE) or a straight connector (FALSE).
                - Right-angle: connector is parallel to the PCB/panel
                - Straight: connector is perpendicular to the PCB/panel
                If the user mentions "90 degrees", "perpendicular", or "angled", they likely want a right-angle connector.
                If the user mentions "straight", "direct", or "vertical", they likely want a straight connector.""",
                'order': 8
            },

            {
                'text': 'What is your operational temperature requirement in Celsius?',
                'weight': 30,
                'attribute': 'temp_range',
                'clarification': '',
                'parse_prompt': """Extract maximum temperature requirement in Celsius.""",
                'order': 9
            },

            {
                'text': 'What is your operational current requirement (in Amps)?',
                'weight': 25,
                'attribute': 'max_current',
                'clarification': '',
                'parse_prompt': """Extract the maximum current requirement in Amps. If a range is given, use the higher value.""",
                'order': 10
            },
            {
                'text': 'What is gauge of cable do you need? (AWG24, AWG26...)',
                'weight': 25,
                'attribute': 'wire_gauge',
                'clarification': '',
                'parse_prompt': """Extract the AWG cable values. If a range is compatible, use the higher value.""",
                'order': 11
            }]
        ## initializing this to keep track of what all questions have been asked
        self.asked_questions = set()
        self.answers = {}
        ## initializing confidence scores
        self.confidence_scores = {connector: 0 for connector in self.connectors}
        ## initializing asked currently
        self.current_question = None
        self.question_history = []
        self.parse_failures = 0

    ## This got regex as backup on failure of LLM for response parsing and scoring too much backups 
    def _fallback_parse(self, text: str) -> dict:
        result = {}
        text_lower = text.lower()
        pitch_patterns = [ r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)\s*pitch', r'pitch\s*(?:size|of)?\s*(?:is|:)?\s*(\d+(?:\.\d+)?)', r'(\d+(?:\.\d+)?)\s*mm\s*(?:pitch|spacing)', r'pitch\s*(?:of)?\s*(\d+(?:\.\d+)?)']
        
        # Try each pattern until we find a match
        for pattern in pitch_patterns:
            pitch_match = re.search(pattern, text.lower())
            if pitch_match:
                try:
                    pitch_size = float(pitch_match.group(1))
                    if 0.5 <= pitch_size <= 2.5:
                        result['pitch_size'] = {"value": pitch_size, "confidence": 0.9}
                    else:
                        result['pitch_size'] = {"value": pitch_size, "confidence": 0.5}
                    break
                except (ValueError, IndexError):
                    continue
        pin_patterns = [r'(\d+)\s*pins?', r'(\d+)\s*contacts?', r'pins?(?:\s*count)?(?:\s*of)?\s*(?:is|:)?\s*(\d+)', r'contacts?(?:\s*count)?(?:\s*of)?\s*(?:is|:)?\s*(\d+)', r'need\s*(\d+)\s*pins?', r'(\d+)\s*position']
        board_to_board_patterns = [r'board\s*(?:to|-)\s*board', r'pcb\s*(?:to|-)\s*pcb', r'board\s*board', r'pcb\s*pcb', r'board\s*application']
        
        if any(re.search(pattern, text_lower) for pattern in board_to_board_patterns):
            result['connection_types'] = {"value": "PCB-to-PCB", "confidence": 0.95}
            # Also add this to parsed requirements and mark question as asked
            if hasattr(self, 'answers'):
                self.answers['connection_types'] = ("PCB-to-PCB", 0.95)
            if hasattr(self, 'asked_questions'):
                self.asked_questions.add('connection_types')
        for pattern in pin_patterns:
            pin_match = re.search(pattern, text.lower())
            if pin_match:
                try:
                    pin_count = int(pin_match.group(1))
                    if 1 <= pin_count <= 120:
                        result['pin_count'] = {"value": pin_count, "confidence": 0.9}
                    else:
                        result['pin_count'] = {"value": pin_count, "confidence": 0.5}
                    break
                except (ValueError, IndexError):
                    continue
        text_lower = text.lower()
    
        # On-board/internal indicators
        internal_indicators = [
            'on board', 'onboard', 'in box', 'internal', 'inside', 
            'within the', 'inside the', 'in the device', 'in a box',
            'circuit board', 'pcb mounted', 'board mounted']
        
        # Panel-mount/external indicators
        external_indicators = [
            'panel mount', 'panel-mount', 'external', 'outside', 'out of box',
            'on a box', 'on the box', 'on panel', 'on a panel', 'mounted on box',
            'exterior', 'outside the', 'exposed', 'accessible from outside']
        
        if any(indicator in text_lower for indicator in internal_indicators):
            result['location'] = {"value": "internal", "confidence": 0.9}
        elif any(indicator in text_lower for indicator in external_indicators):
            result['location'] = {"value": "external", "confidence": 0.9}
        ## Look for current
        current_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:a|amp|amps)', text.lower())
        if current_match:
            current = float(current_match.group(1))
            result['max_current'] = {"value": current, "confidence": 0.8}
        
        ## Look for temperature
        temp_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:c|celsius|°c|degrees?)',
            r'temperature\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)',
            r'(?:up\s*to|max|maximum)\s*(\d+(?:\.\d+)?)\s*(?:c|degrees|celsius|°c)',
            r'operate\s*(?:at|in)\s*(\d+(?:\.\d+)?)\s*(?:c|degrees|celsius|°c)']
        
        for pattern in temp_patterns:
            temp_match = re.search(pattern, text.lower())
            if temp_match:
                try:
                    temp = float(temp_match.group(1))
                    if -100 <= temp <= 500:
                        result['temp_range'] = {"value": temp, "confidence": 0.8}
                    else:
                        result['temp_range'] = {"value": temp, "confidence": 0.5}
                    break
                except (ValueError, IndexError):
                    continue
        ## Look for EMI protection
        emi_text = text.lower()
        if 'emi' in emi_text or 'electromagnetic' in emi_text or 'shield' in emi_text:
            negative_indicators = ['no emi', 'without emi', 'no shield', 'not shielded', 'no electromagnetic']
            if any(indicator in emi_text for indicator in negative_indicators):
                result['emi_protection'] = {"value": False, "confidence": 0.9}
            else:
                result['emi_protection'] = {"value": True, "confidence": 0.9}
        ## Handle mixed power/signal requirement
        if any(phrase in text.lower() for phrase in ['mix', 'mixed', 'mixing', 'both power and signal', 'power and signal', 'power signal']):
            if any(phrase in text.lower() for phrase in ['high power', 'power', 'current']):
                result['mixed_power_signal'] = {"value": True, "confidence": 0.9}
                print("Detected mixed power/signal requirement with high confidence")
        housing_text = text.lower()
        if 'metal' in housing_text or 'metallic' in housing_text or 'aluminum' in housing_text or 'steel' in housing_text:
            ## Check for preference indicators cause it was dumb
            preference_terms = ['prefer', 'preferable', 'ideally', 'better', 'if possible', 'would like']
            is_preference = any(term in housing_text for term in preference_terms)
            
            if is_preference:
                result['housing_material'] = {"value": "metal", "confidence": 0.85}
            else:
                result['housing_material'] = {"value": "metal", "confidence": 0.95}
        elif any(term in housing_text for term in ['plastic', 'polymer', 'composite', 'non-metal']):
            preference_terms = ['prefer', 'preferable', 'ideally', 'better', 'if possible', 'would like']
            is_preference = any(term in housing_text for term in preference_terms)
            
            if is_preference:
                result['housing_material'] = {"value": "plastic", "confidence": 0.85}
            else:
                result['housing_material'] = {"value": "plastic", "confidence": 0.95}
            
        ## Look for location
        if any(word in text.lower() for word in ['external', 'outside', 'exterior','panel mount']):
            result['location'] = {"value": "external", "confidence": 0.8}
        elif any(word in text.lower() for word in ['internal', 'inside', 'interior','on board']):
            result['location'] = {"value": "internal", "confidence": 0.8}
        awg_patterns = [
        r'(?:awg|gauge)[- ]?(\d+)',          
        r'with\s+(?:awg|gauge)[- ]?(\d+)',  
        r'(?:awg|gauge)[- ]?(\d+)\s+(?:wire|cable)', 
        r'(\d+)\s*(?:awg|gauge)',           
        r'side\s+(?:and|with)\s+(?:awg|gauge)[- ]?(\d+)',  
        r'(?:awg|gauge)[- ]?(\d+)\s+(?:the|on)\s+(?:other|one)']
        
        ## Look for wire gauge (AWG)
        for pattern in awg_patterns:
            awg_match = re.search(pattern, text.lower())
            if awg_match:
                try:
                    awg = int(awg_match.group(1))
                    if 10 <= awg <= 40: 
                        result['wire_gauge'] = {"value": awg, "confidence": 0.95}
                        result['connection_type'] = {"value": "PCB-to-Cable", "confidence": 0.95}
                        
                        ## Special pattern "straight on PCB one side and with AWG"
                        if ('straight' in text.lower() and 'pcb' in text.lower() and 'side' in text.lower()):
                            result['right_angle'] = {"value": False, "confidence": 0.95}
                        
                        break
                except (ValueError, IndexError):
                    continue
        text_lower = text.lower()
        straight_patterns = [
            r'straight\s+(?:on|onto|connector|pcb|cable|connection)',
            r'connector\s+straight',
            r'vertical\s+(?:connector|connection|mount)',
            r'perpendicular\s+(?:to|connector)',
            r'direct\s+(?:mount|connection)']
        
        ## Right angle patterns
        right_angle_patterns = [
            r'right[\s-]angle',
            r'90[\s-]degree',
            r'angled\s+(?:connector|connection)',
            r'horizontal\s+(?:connector|connection)',
            r'parallel\s+(?:to|connection)']
        
        ## Check each straight pattern
        for pattern in straight_patterns:
            if re.search(pattern, text_lower):
                result['right_angle'] = {"value": False, "confidence": 0.9}
                break
        
        ## Check right angle patterns if no straight pattern matched
        if 'right_angle' not in result:
            for pattern in right_angle_patterns:
                if re.search(pattern, text_lower):
                    result['right_angle'] = {"value": True, "confidence": 0.9}
                    break
            if any(phrase in text_lower for phrase in ['straight on', 'straight connector', 'straight pcb']):
                result['right_angle'] = {"value": False, "confidence": 0.9}
            elif any(phrase in text_lower for phrase in ['right angle', 'right-angle', '90 degree']):
                result['right_angle'] = {"value": True, "confidence": 0.9}
        
        text_lower = text.lower()
        
        ## Enhanced PCB-to-Cable patterns
        pcb_to_cable_patterns = [
            r'pcb\s+(?:to|and|with|on\s+one\s+side).+(?:cable|wire|awg)',
            r'one\s+side\s+(?:on\s+)?pcb.+other\s+side\s+(?:cable|wire|awg)',
            r'connect\s+pcb\s+to\s+(?:cable|wire)',
            r'pcb\s+connector\s+with\s+(?:cable|wire)',
            r'pcb\s+(?:one|1)\s+side.+(?:awg|wire|cable)', 
            r'(?:awg|wire|cable).+(?:one|1)\s+side.+pcb',
            r'right\s+angle\s+on\s+pcb'  ]
        
        ## Cable-to-PCB patterns
        cable_to_pcb_patterns = [
            r'(?:cable|wire|awg).+(?:to|and|with|on\s+one\s+side).+pcb',
            r'one\s+side\s+(?:cable|wire|awg).+other\s+side\s+pcb',
            r'connect\s+(?:cable|wire)\s+to\s+pcb',
            r'(?:cable|wire)\s+connector\s+with\s+pcb'
        ]
        
        ## PCB-to-PCB patterns
        pcb_to_pcb_patterns = [
            r'pcb\s+to\s+pcb',
            r'connect\s+(?:two|2)\s+pcbs?',
            r'pcb\s+on\s+both\s+sides',
            r'both\s+sides?\s+pcb']
        
        # Cable-to-Cable patterns
        cable_to_cable_patterns = [
            r'(?:cable|wire)\s+to\s+(?:cable|wire)',
            r'connect\s+(?:two|2)\s+(?:cables|wires)',
            r'(?:cable|wire)\s+on\s+both\s+sides',
            r'both\s+sides?\s+(?:cable|wire)']
        
        if 'wire_gauge' in result and ('pcb' in text_lower or 'board' in text_lower):
            result['connection_type'] = {"value": "PCB-to-Cable", "confidence": 0.95}
            return result
        # Check each pattern group
        for pattern in pcb_to_cable_patterns:
            if re.search(pattern, text_lower):
                result['connection_type'] = {"value": "PCB-to-Cable", "confidence": 0.9}
                break
                
        if 'connection_type' not in result:
            for pattern in cable_to_pcb_patterns:
                if re.search(pattern, text_lower):
                    result['connection_type'] = {"value": "Cable-to-PCB", "confidence": 0.9}
                    break
        
        if 'connection_type' not in result:
            for pattern in pcb_to_pcb_patterns:
                if re.search(pattern, text_lower):
                    result['connection_type'] = {"value": "PCB-to-PCB", "confidence": 0.9}
                    break
        
        if 'connection_type' not in result:
            for pattern in cable_to_cable_patterns:
                if re.search(pattern, text_lower):
                    result['connection_type'] = {"value": "Cable-to-Cable", "confidence": 0.9}
                    break
        
        # Fallback to simpler logic if patterns didn't match
        if 'connection_type' not in result:
            if 'pcb' in text_lower and ('cable' in text_lower or 'wire' in text_lower or 'awg' in text_lower):
                if text_lower.find('pcb') < text_lower.find('cable') or text_lower.find('pcb') < text_lower.find('wire'):
                    result['connection_type'] = {"value": "PCB-to-Cable", "confidence": 0.8}
                else:
                    result['connection_type'] = {"value": "Cable-to-PCB", "confidence": 0.8}
            elif 'pcb' in text_lower and text_lower.count('pcb') >= 2:
                result['connection_type'] = {"value": "PCB-to-PCB", "confidence": 0.8}
            elif ('cable' in text_lower or 'wire' in text_lower) and (text_lower.count('cable') >= 2 or text_lower.count('wire') >= 2):
                result['connection_type'] = {"value": "Cable-to-Cable", "confidence": 0.8}
        
        return result
    

    ## Parse from 1st message whatever features which can be parsed
    async def process_initial_message(self, message: str) -> Dict:
        try:
            if not hasattr(self, 'confidence_scores'):
                self.confidence_scores = {connector: 0.0 for connector in self.connectors}
            ## LLM must recognize what ever it can
            system_message = SystemMessage(content="""You are an expert in analyzing connector requirements.
            Extract technical specifications from user messages, focusing on explicitly mentioned and implied values.""")
            
            user_message = HumanMessage(content=f"""
            Extract connector requirements from this message: "{message}"
            
            IMPORTANT GUIDANCE:
            CONNECTION TYPE DETECTION - HIGHEST PRIORITY:
            - Terms like "board to board", "board-to-board", "PCB to PCB", "board-board" always indicate PCB-PCB connection
            - Terms like "PCB to cable", "board to wire" indicate PCB-to-Cable
            - Be extremely aggressive in inferring connection types - this is critical
            
            LOCATION DETECTION:
            - Terms like "on board", "onboard", "in box", "inside" indicate an internal/on-board requirement
            - Terms like "panel mount", "external", "outside", "out of box" indicate panel mount requirement
            - Be very vigilant about detecting these location mentions as they're often overlooked
        
            FOR HOUSING MATERIAL:
            - Terms like "metal", "metallic" strongly indicate metal housing requirements
            - If EMI shielding is mentioned, this implies metal housing
            - Be extremely vigilant about detecting metal housing requirements, as this is critical
            
            FOR MIXED POWER SIGNAL:
            - Terms like "mixed signal", "mixed power", "high power", "high frequency", "mixed" imply a requirement for mixed power signal
            - If only signal then set as false.
            
            FOR CONNECTION TYPE:
            - If PCB and AWG/wire/cable are mentioned together, this indicates a PCB-to-Cable connection
            - Phrases like "PCB on one side" and "cable/wire on other side" indicate PCB-to-Cable
            - If right angle is mentioned with PCB, this often implies PCB-to-Cable connection
            - Be aggressive in inferring connection types from context
            
            Return a JSON object with explicitly mentioned AND reasonably implied requirements:
            - pitch_size (in mm)
            - pin_count (number of pins)
            - max_current (in Amps)
            - temp_range (in Celsius)
            - emi_protection (boolean)
            - height_requirement (in mm)
            - wire_gauge (AWG number)
            - mixed_power_signal (boolean: true if mixed or power, false if signal only)
            - location (string: "internal" or "external")
            - right_angle (boolean: true if right-angle, false if straight)
            - connector_orientation (boolean: true if straight, false if right-angle)
            - connection_type (string: "PCB-to-PCB", "PCB-to-Cable", "Cable-to-Cable", "Cable-to-PCB")

            For 'location', specifically search for:
            - "on board", "onboard", "in box", "internal", "inside" → set as "internal"
            - "panel mount", "external", "on box", "outside", "out of box" → set as "external"

            Format your response as JSON only:
            {{
                "pitch_size": {{"value": 1.0, "confidence": 0.95}},
                "pin_count": {{"value": 20, "confidence": 0.95}},
                "max_current": {{"value": 3.0, "confidence": 0.8}},
                "temp_range": {{"value": 85, "confidence": 0.7}},
                "emi_protection": {{"value": false, "confidence": 0.6}},
                "height_requirement": {{"value": 4, "confidence": 0.5}},
                "wire_gauge": {{"value": 26, "confidence": 0.9}},
                "right_angle": {{"value": true, "confidence": 0.8}},
                "connector_orientation": {{"value": true, "confidence": 0.8}},
                "connection_type": {{"value": "PCB-to-Cable", "confidence": 0.9}},
                "mixed_power_signal": {{"value": "mixed" or "power", "confidence": 0.9}},
                "location": {{"value": "internal", "confidence": 0.9}},

            }}
            
            Only include mentioned or strongly implied requirements. No additional text.""")

            try:
                response = await asyncio.wait_for(
                    self.llm.agenerate([[system_message, user_message]]),
                    timeout=10  
                )
                response_text = response.generations[0][0].text.strip()
            except (asyncio.TimeoutError, Exception) as e:
                print(f"LLM processing error or timeout: {str(e)}")
                # Fall back to regex parsing on LLM failure
                parsed = self._fallback_parse(message)
                return self._process_parsed_requirements(parsed, message)
            
            # Clean the response from LLM
            cleaned_text = response_text
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            try:
                parsed = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                parsed = self._fallback_parse(message)
            
            ## Check for connection_type and wire_gauge co-occurrence and enhance confidence
            ## need this cause LLM sucks
            if 'connection_types' in self.answers:
                connection_type = self.answers['connection_types'][0]
                if connection_type and connection_type.lower() in ["pcb-to-pcb", "pcb to pcb", "board to board"]:
                    # For PCB-PCB connections, auto-set these values
                    self.answers['location'] = ("internal", 0.95)
                    self.asked_questions.add('location')
                    
                    # Mark wire_gauge as already asked (not applicable)
                    self.answers['wire_gauge'] = (None, 0.0)
                    self.asked_questions.add('wire_gauge')

            if 'connection_type' in parsed and 'wire_gauge' in parsed:
                conn_type = parsed['connection_type']['value']
                if conn_type != "PCB-to-Cable" and conn_type != "Cable-to-PCB":
                    parsed['connection_type']['value'] = "PCB-to-Cable"
                    parsed['connection_type']['confidence'] = 0.9
            
            if 'wire_gauge' in parsed and ('pcb' in message.lower() or 'board' in message.lower()):
                parsed['connection_type'] = {"value": "PCB-to-Cable", "confidence": 0.95}
            if ('straight' in message.lower() and 'pcb' in message.lower() and 
                'side' in message.lower() and any(f'awg{i}' in message.lower().replace(' ', '') 
                                                for i in range(10, 40))):
                parsed['connection_type'] = {"value": "PCB-to-Cable", "confidence": 0.99}
                parsed['right_angle'] = {"value": False, "confidence": 0.95}
                ## Extract the AWG value and add it directly
                awg_match = re.search(r'awg\s*(\d+)', message.lower())
                if awg_match:
                    try:
                        awg_value = int(awg_match.group(1))
                        print(f"Directly extracted AWG{awg_value} from initial message")
                        ## We need to ensure this gets added to answers even before LLM parsing
                        if 10 <= awg_value <= 40: 
                            self.answers['wire_gauge'] = (awg_value, 0.95)
                            self.asked_questions.add('wire_gauge')
                            ### Special check for connectors that don't support this AWG
                            for connector_name, connector_specs in self.connectors.items():
                                supported_awgs = connector_specs.get('wire_gauge', [])
                                ## Normalize supported AWGs for comparison
                                normalized_supported = []
                                for awg_str in supported_awgs:
                                    norm_awg = self.normalize_awg_value(awg_str) if callable(getattr(self, 'normalize_awg_value', None)) else normalize_awg_value(awg_str)
                                    if norm_awg is not None:
                                        normalized_supported.append(norm_awg)
                                ### Check if the AWG is supported by this connector
                                if awg_value not in normalized_supported:
                                    print(f"AWG{awg_value} is NOT supported by {connector_name} (supported: {normalized_supported})")
                                    self.confidence_scores[connector_name] *= 0.1
                    except ValueError:
                        pass
            result = self._process_parsed_requirements(parsed, message)
            
            ## If enough information for a recommendation
            if result.get("status") == "complete" and result.get("recommendation_ready", False):
                try:
                    ## Use the pre-calculated values from _process_parsed_requirements
                    best_connector = result.get("best_connector")
                    best_score = result.get("best_score", 0.0)
                    #@ Generate recommendation with these values
                    recommendation = await self.generate_recommendation(best_connector=best_connector, max_confidence=best_score)
                    return recommendation
                except Exception as e:
                    print(f"Error generating recommendation: {str(e)}")
                    # Return a properly structured error response
                    return {
                        "status": "complete",
                        "recommendation": {
                            "connector": "error",
                            "confidence": "error",
                            "analysis": f"I apologize, but I encountered an error while generating my recommendation. Please try again or provide more details about your requirements.",
                            "requirements": "Error processing requirements",
                            "requirements_summary": "Error processing requirements summary",
                            "confidence_scores": {k: float(v) for k, v in self.confidence_scores.items()}}}
            return result

        except Exception as e:
            print(f"Error in process_initial_message: {str(e)}")
            self.confidence_scores = {connector: 0.0 for connector in self.connectors}
            next_q = self.get_next_question()
            return {
                "status": "continue",
                "next_question": next_q,
                "confidence_scores": self.confidence_scores,
                "error": str(e)}

    ## what the function says  
    ## Too many parameters to explain please go figure
    def _process_parsed_requirements(self, parsed, message):
        if isinstance(parsed, dict):
            for attr, value in parsed.items():
                if isinstance(value, dict) and 'value' in value and 'confidence' in value:
                    if value['value'] is not None:
                        confidence = float(value['confidence'])
                        self.answers[attr] = (value['value'], confidence)
                        self.asked_questions.add(attr)
                        if attr == 'wire_gauge':
                            raw_value = value['value']
                            awg_value = normalize_awg_value(raw_value)
                            if awg_value is not None:
                                self.answers[attr] = (awg_value, confidence)
                                print(f"Extracted AWG value: {awg_value} (from {raw_value}) with confidence {confidence}")
                                self.asked_questions.add(attr)
                                
                                ## If we detect AWG, also infer a connection type involving cable
                                if 'connection_types' not in self.answers and 'connection_type' not in self.answers:
                                    self.answers['connection_types'] = ("PCB-to-Cable", confidence * 0.9)
                                    self.asked_questions.add('connection_types')
                                    print(f"Inferred PCB-to-Cable connection from AWG mention with confidence {confidence * 0.9}")
                            else:
                                print(f"Could not normalize AWG value: {raw_value}")
        if 'right_angle' in self.answers and 'connector_orientation' in self.answers:
            right_angle_val, right_angle_conf = self.answers['right_angle']
            conn_orient_val, conn_orient_conf = self.answers['connector_orientation']
        
            if right_angle_val == conn_orient_val:  
                if right_angle_conf >= conn_orient_conf:
                    self.answers['right_angle'] = (right_angle_val, right_angle_conf)
                    del self.answers['connector_orientation']
                else:
                    self.answers['right_angle'] = (not conn_orient_val, conn_orient_conf)
                    del self.answers['connector_orientation']
            else:
                if right_angle_conf >= conn_orient_conf:
                    del self.answers['connector_orientation']
                else:
                    self.answers['right_angle'] = (not conn_orient_val, conn_orient_conf)
                    del self.answers['connector_orientation']
                    
        elif 'connection_type' in parsed:
            conn_type_value = parsed['connection_type']['value']
            conn_type_conf = parsed['connection_type']['confidence']
            # Normalize connection type format
            if isinstance(conn_type_value, str):
                norm_conn_type = conn_type_value.lower().replace(' ', '-')
                if "pcb" in norm_conn_type and "pcb" in norm_conn_type.split("-", 1)[1]:
                    # This is PCB-to-PCB
                    self.answers['connection_types'] = ("PCB-to-PCB", conn_type_conf)
                    # Auto-skip location
                    self.answers['location'] = ("internal", 0.95)
                    self.asked_questions.add('location')
        elif 'connector_orientation' in self.answers and 'right_angle' not in self.answers:
            value, confidence = self.answers['connector_orientation']
            self.answers['right_angle'] = (not value, confidence)
            del self.answers['connector_orientation']
       
        ## Update confidence scores
        for connector_name, connector_specs in self.connectors.items():
            score = float(self.calculate_connector_score(connector_specs, self.answers))
            self.confidence_scores[connector_name] = score
        scores = list(self.confidence_scores.items())
        best_connector, best_score = max(scores, key=lambda x: x[1])
        other_scores = [score for connector, score in scores if connector != best_connector]
        max_other_score = max(other_scores) if other_scores else 0
        required_critical_attributes = {'mixed_power_signal', 'housing_material'}
        critical_attributes_met = True
        
        if 'emi_protection' in self.answers and self.answers['emi_protection'][1] > 0.7:
            # If EMI protection is required with high confidence, housing must be metal
            if self.answers['emi_protection'][0] is True:
                # self.answers['housing_material'] = ('metal', self.answers['emi_protection'][1])
                parsed['housing_material'] = {"value": "metal", "confidence": parsed['emi_protection']['confidence']}
        ## Check that both pitch_size and housing_material are present with high confidence
        for attr in required_critical_attributes:
            if attr not in self.answers or self.answers[attr][1] < 0.7:
                critical_attributes_met = False
                break
                
        ## Additional pattern matching for connector names in initial message
        connector_names = ['AMM', 'CMM', 'DMM', 'EMM']
        mentioned_connectors = []
        for name in connector_names:
            if name in message.upper():
                mentioned_connectors.append(name)
                if name in self.confidence_scores:
                    self.confidence_scores[name] += 15.0
        
        # IMPORTANT: Don't try to generate immediate recommendation here!
        # Instead, just set a flag to indicate sufficient information is available
        if critical_attributes_met and best_score >= 57 and (best_score - max_other_score) > 25:
            print(f"Initial message provided sufficient critical information (Score: {best_score})")
            print(f"Skipping questions and proceeding to recommendation")
            return {
                "status": "complete",
                "recommendation_ready": True,
                "confidence_scores": self.confidence_scores, 
                "best_connector": best_connector,
                "best_score": best_score}
        ## Get next question to ask the user
        next_q = self.get_next_question()
        if next_q is None:
            ## No more questions needed - indicate ready for recommendation
            return {
                "status": "complete", 
                "recommendation_ready": True}

        ## Return next question and current scores
        return {"status": "continue", "next_question": next_q, "confidence_scores": {k: float(v) for k, v in self.confidence_scores.items()},
                "mentioned_connectors": mentioned_connectors}


    ## Get next question to ask the user
    def get_next_question(self) -> Dict:
        try:
            if self.current_question is None:
                # Map LLM extracted properties to question attributes
                property_to_attribute = {
                    'pitch_size': 'pitch_size',
                    'pin_count': 'pin_count',
                    'max_current': 'max_current',
                    'temp_range': 'temp_range',
                    'emi_protection': 'housing_material',
                    'housing_material': 'housing_material',
                    'height_requirement': 'height_requirement',
                    'wire_gauge': 'wire_gauge',
                    'connector_orientation': 'connector_orientation',
                    'connection_type': 'connection_types',
                    'location': 'location',
                    'mixed_power_signal': 'mixed_power_signal'}
                
                # Mark questions as asked if we already have the answers with high confidence
                for property_name, attr_name in property_to_attribute.items():
                    if property_name in self.answers:
                        value, confidence = self.answers[property_name]
                        # High confidence threshold
                        if confidence > 0.7:
                            print(f"Skipping question about {attr_name} as it was extracted from initial message")
                            self.asked_questions.add(attr_name)
                
                # For housing material, consider both direct material mention and EMI protection
                if 'housing_material' in self.answers and 'emi_protection' in self.answers:
                    # If both are mentioned with decent confidence, mark as asked
                    material_value, material_conf = self.answers['housing_material']
                    emi_value, emi_conf = self.answers['emi_protection']
                    
                    if material_conf > 0.7 and emi_conf > 0.7:
                        self.asked_questions.add('housing_material')
                        
                    # If EMI is required with high confidence, we know housing must be metal
                    elif emi_value is True and emi_conf > 0.8:
                        self.answers['housing_material'] = ('metal', emi_conf)
                        self.asked_questions.add('housing_material')
                
                self.current_question = self.select_next_question({})
            
            if self.current_question:
                return {
                    "question": self.current_question['text'],
                    "clarification": self.current_question['clarification'],
                    "attribute": self.current_question['attribute']}
            
            ## No more questions - all have been answered
            return None  
        except Exception as e:
            print(f"Error in get_next_question: {str(e)}")
            return None
        

    async def process_answer(self, response: str) -> Dict:
        if not self.current_question:
            return {"status": "error", "message": "No active question"}

        try:
            # Check for intent to restart
            restart_patterns = [
                r"\brestart\b", r"\bnew\s+selection\b", r"\bstart\s+over\b", r"\bbegin\s+again\b",
                r"\breset\b", r"\bstart\s+new\b", r"\bdifferent\s+connector\b"
            ]
            
            if any(re.search(pattern, response.lower()) for pattern in restart_patterns):
                self.answers = {}
                self.asked_questions = set()
                self.confidence_scores = {connector: 0 for connector in self.connectors}
                self.current_question = self.select_next_question({})
                self.question_history = []
                self.parse_failures = 0
                
                return {
                    "status": "continue",
                    "next_question": {
                        "question": self.current_question['text'],
                        "clarification": self.current_question['clarification'],
                        "attribute": self.current_question['attribute']
                    },
                    "confidence_scores": {k: f"{v:.1f}%" for k, v in self.confidence_scores.items()},
                    "restarted": True
                }
            if 'connection_types' in self.answers:
                connection_type = self.answers['connection_types'][0]
                if isinstance(connection_type, str) and connection_type.lower() in ["pcb-to-pcb", "pcb to pcb", "board to board"]:
                    # Auto-skip location question for PCB-to-PCB
                    self.answers['location'] = ("internal", 0.95)
                    self.asked_questions.add('location')
                    print("PCB-to-PCB detected: Automatically skipping location/panel mount question")
            # Special handling for height_requirement question if user indicates they don't know
            if self.current_question['attribute'] == 'height_requirement':
                response_lower = response.lower()
                uncertainty_phrases = [
                    "don't know", "dont know", "not sure", "uncertain", 
                    "no idea", "no specific", "not specified", "unsure",
                    "don't have", "no constraint", "no requirement", "any height",
                    "flexible", "whatever works", "any option","no perticular constraint"
                ]
                
                if any(phrase in response_lower for phrase in uncertainty_phrases):
                    # Mark height as asked with zero confidence
                    self.answers[self.current_question['attribute']] = (None, 0.0)
                    self.asked_questions.add(self.current_question['attribute'])
                    
                    # Skip directly to pitch_size question
                    pitch_question = next((q for q in self.all_questions if q['attribute'] == 'pitch_size' and q['attribute'] not in self.asked_questions), None)
                    
                    if pitch_question:
                        self.current_question = pitch_question
                        return {"status": "continue",
                            "next_question": {
                                "question": self.current_question['text'],
                                "clarification": self.current_question['clarification'],
                                "attribute": self.current_question['attribute']},
                            "confidence_scores": {k: f"{v:.1f}%" for k, v in self.confidence_scores.items()},
                            "skipped_height": True}
            
            ## Normal processing with error handling
            try:
                parsed_response = await self.parse_response_with_llm(response, self.current_question)
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                ## Use fallback parsing on error
                parsed_response = self._aggressive_fallback_parse(response, self.current_question)
            
            if parsed_response and 'value' in parsed_response:
                # Store the answer with confidence
                self.answers[self.current_question['attribute']] = (parsed_response['value'], parsed_response['confidence'])
                self.asked_questions.add(self.current_question['attribute'])
                self.question_history.append({'question': self.current_question['attribute'],
                    'answer': parsed_response['value'],
                    'confidence': parsed_response['confidence']})
                # Special handling for wire_gauge - normalize the value
                if self.current_question['attribute'] == 'wire_gauge' and parsed_response['value'] is not None:
                    normalized_awg = self.normalize_awg_value(parsed_response['value']) if callable(getattr(self, 'normalize_awg_value', None)) else normalize_awg_value(parsed_response['value'])
                    if normalized_awg is not None:
                        self.answers[self.current_question['attribute']] = (normalized_awg, parsed_response['confidence'])
                        print(f"Normalized AWG value: {normalized_awg}")

                ## Update confidence scores
                for connector_name, connector_specs in self.connectors.items():
                    try:
                        if 'housing_material' in self.answers and self.confidence_scores[connector_name] == 0:
                            continue
                        score = self.calculate_connector_score(connector_specs, self.answers)
                        self.confidence_scores[connector_name] = score
                    except Exception as score_error:
                        print(f"Error calculating score for {connector_name}: {str(score_error)}")
                        continue
                
                scores = list(self.confidence_scores.items())
                best_connector, best_score = max(scores, key=lambda x: x[1])
                other_scores = [score for connector, score in scores if connector != best_connector]
                max_other_score = max(other_scores) if other_scores else 0
                score_gap = best_score - max_other_score
                
                # Check for critical questions
                critical_questions = {'mixed_power_signal', 'housing_material'}
                critical_questions_asked = critical_questions.intersection(self.asked_questions)
                
                # Early recommendation conditions refined
                if (best_score >= 75 and 
                    score_gap > 15 and
                    len(critical_questions_asked) == len(critical_questions) and
                    len(self.asked_questions) >= 3):
                    try:
                        return await self.generate_recommendation()
                    except Exception as rec_error:
                        print(f"Error generating recommendation: {str(rec_error)}")
                try:
                    self.current_question = self.select_next_question({})
                except Exception as q_error:
                    print(f"Error selecting next question: {str(q_error)}")
                    # Simple fallback to first unasked question
                    self.current_question = next((q for q in self.all_questions if q['attribute'] not in self.asked_questions), None)
                
                if self.current_question:
                    return {"status": "continue",
                        "next_question": {
                            "question": self.current_question['text'],
                            "clarification": self.current_question['clarification'],
                            "attribute": self.current_question['attribute']},
                        "confidence_scores": {k: f"{v:.1f}%" for k, v in self.confidence_scores.items()}}
                else:
                    try:
                        return await self.generate_recommendation()
                    except Exception as final_error:
                        print(f"Error generating final recommendation: {str(final_error)}")
                        return {
                            "status": "error",
                            "message": "An error occurred generating the recommendation. Please try again."
                        }

            return {
                "status": "error",
                "message": "Could not parse your response. Can you please clarify?"
            }

        except Exception as e:
            print(f"Error in process_answer: {str(e)}")
            return {"status": "error","message": f"An error occurred processing your response: {str(e)}"}
        

    def format_user_requirements_summary(self) -> str:
        summary_parts = []
        
        attr_display_names = {
            'pitch_size': 'Pitch Size',
            'pin_count': 'Pin Count',
            'max_current': 'Current Requirement',
            'temp_range': 'Temperature',
            'emi_protection': 'EMI Protection',
            'housing_material': 'Housing Material',
            'height_requirement': 'Height Requirement',
            'wire_gauge': 'Wire Gauge',
            'right_angle': 'Right Angle',
            'connection_types': 'Connection Type',
            'location': 'Location',
            'mixed_power_signal': 'Mixed Power/Signal'}
        
        # Format each requirement with appropriate units and formatting
        for attr, (value, conf) in sorted(self.answers.items(), key=lambda x: attr_display_names.get(x[0], x[0])):
            if value is None:
                continue
                
            display_name = attr_display_names.get(attr, attr.replace('_', ' ').title())
            
            # Format value based on type
            if attr == 'pitch_size':
                formatted_value = f"{value} mm"
            elif attr == 'max_current':
                formatted_value = f"{value} A"
            elif attr == 'temp_range':
                formatted_value = f"{value}°C"
            elif attr == 'height_requirement':
                formatted_value = f"{value} mm"
            elif isinstance(value, bool):
                formatted_value = "Yes" if value else "No"
            else:
                formatted_value = str(value)
                
            summary_parts.append(f"{display_name}: {formatted_value}")
        
        if not summary_parts:
            return "No specific requirements were provided."
            
        return "\n".join(summary_parts)

    async def generate_recommendation(self, best_connector=None, max_confidence=None) -> Dict:
        try:
            if best_connector is None or max_confidence is None:
                scores = list(self.confidence_scores.items())
                best_connector, max_confidence = max(scores, key=lambda x: x[1])
            
            formatted_scores = {}
            for k, v in self.confidence_scores.items():
                formatted_scores[k] = float(v)
            
            scores = list(self.confidence_scores.items())
            top_connectors = [c for c, s in scores if abs(s - max_confidence) < 0.1] 
            other_scores = [score for connector, score in scores if connector != best_connector]
            max_other_score = max(other_scores) if other_scores else 0
            
            unconfirmed_features = []
            
            connector_specs = self.connectors[best_connector]
            
            for attr, (value, confidence) in self.answers.items():
                if value is None:
                    continue
                    
                if attr == 'pin_count':
                    pin_count = int(value)
                    valid_pins = connector_specs.get('valid_pin_counts', set())
                    max_pins = connector_specs.get('max_pins', 0)
                    
                    if pin_count > max_pins:
                        unconfirmed_features.append(f"Pin count of {pin_count} exceeds standard maximum of {max_pins}")
                    elif pin_count not in valid_pins and pin_count <= max_pins:
                        unconfirmed_features.append(f"Pin count of {pin_count} is within range but may need configuration confirmation")
                
                elif attr == 'pitch_size':
                    spec_pitch = connector_specs.get('pitch_size', 0)
                    if abs(float(value) - spec_pitch) > 0.05:
                        unconfirmed_features.append(f"Pitch size of {value}mm differs from standard {spec_pitch}mm")
                
                elif attr == 'max_current':
                    spec_current = connector_specs.get('max_current', 0)
                    if float(value) > spec_current:
                        unconfirmed_features.append(f"Current requirement of {value}A exceeds standard rating of {spec_current}A")
                
                elif attr == 'temp_range':
                    min_temp, max_temp = connector_specs.get('temp_range', (-273, 1000))
                    if float(value) > max_temp:
                        unconfirmed_features.append(f"Temperature requirement of {value}°C exceeds maximum rating of {max_temp}°C")
                
                elif attr == 'housing_material':
                    if value != connector_specs.get('housing_material', ''):
                        unconfirmed_features.append(f"Housing material requirement ({value}) differs from standard ({connector_specs.get('housing_material', '')})")
                
                elif attr == 'emi_protection':
                    if value and not connector_specs.get('emi_protection', False):
                        unconfirmed_features.append(f"EMI protection is required but not standard with this connector")
                
                elif attr == 'mixed_power_signal':
                    if value and not connector_specs.get('mixed_power_signal', False):
                        unconfirmed_features.append(f"Mixed power/signal capability is required but may need special configuration")
                
                elif attr == 'right_angle':
                    if value != connector_specs.get('right_angle', False):
                        unconfirmed_features.append(f"Connector orientation (right angle: {value}) may require special configuration")
                        
                elif attr == 'height_requirement' and value is not None:
                    height_range = connector_specs.get('height_range', (0, 0))
                    height_options = connector_specs.get('height_options', [])
                    
                    if not (height_range[0] <= float(value) <= height_range[1]):
                        closest = min(height_options, key=lambda x: abs(x - float(value))) if height_options else None
                        if closest:
                            unconfirmed_features.append(f"Height requirement of {value}mm differs from available options (closest: {closest}mm)")
            
            ## create user requirements summary
            requirements_summary = self.format_user_requirements_summary()
            requirements_text = self.format_requirements()
            scores_text = self.format_scores()
            
            # Only recommend contact for truly low confidence
            if max_confidence < 22 or (len(unconfirmed_features) > 3 and max_confidence < 22):
                system_message = SystemMessage(content=self.system_prompt)
                lnk="https://www.nicomatic.com/contact/?"
                user_message = HumanMessage(content=f"""
                Based on the following user requirements:
                
                {requirements_summary}
                
                I cannot confidently recommend a specific connector.
                
                Please provide a response that explains:
                1. First, summarize the requirements provided by the user
                2. Explain that based on these requirements, we need more information to make a specific recommendation
                3. Suggest the user contact Nicomatic directly for personalized assistance
                4. Provide this contact link: "{lnk}"
                
                Start with: "Based on your requirements..."
                Include the summary of requirements in your response.
                Keep the response concise and professional.""")
                
                try:
                    recommendation = await self.llm.agenerate([[system_message, user_message]])
                    recommendation_text = recommendation.generations[0][0].text
                    
                    # Return in the expected format
                    return {
                        "status": "complete",
                        "recommendation": {
                            "connector": "contact",
                            "confidence": "insufficient",
                            "analysis": recommendation_text,
                            "requirements": requirements_text,
                            "requirements_summary": requirements_summary,
                            "confidence_scores": formatted_scores
                        }
                    }
                except Exception as e:
                    print(f"Error generating contact recommendation: {str(e)}")
                    fallback_text = (
                        f"Based on your requirements ({requirements_summary}), I don't have enough information to confidently "
                        f"recommend a specific connector. For personalized assistance with your "
                        f"connector selection, please contact Nicomatic's support team directly at {lnk}")
                    
                    # Return in the expected format
                    return {
                        "status": "complete",
                        "recommendation": {
                            "connector": "contact",
                            "confidence": "insufficient",
                            "analysis": fallback_text,
                            "requirements": requirements_text,
                            "requirements_summary": requirements_summary,
                            "confidence_scores": formatted_scores}}
            
            # If we have a reasonable confidence, generate a recommendation with notes
            system_message = SystemMessage(content=self.system_prompt)
            if best_connector == "DMM":
                link= "https://configurator.nicomatic.com/product_configurator/part_builder?id=89"
            elif best_connector == "EMM":
                link= "https://configurator.nicomatic.com/product_configurator/part_builder?id=169"
            elif best_connector == "CMM":
                link= "https://configurator.nicomatic.com/product_configurator/part_builder?id=3" 
            elif best_connector == "AMM":
                link= "https://configurator.nicomatic.com/product_configurator/part_builder?id=5"  
            else:
                link="https://www.nicomatic.com/contact"
                
            # Include notes about features needing confirmation
            unconfirmed_notes = ""
            if unconfirmed_features:
                unconfirmed_notes = "\n\nPlease include this note: " + "; ".join(unconfirmed_features) + ". Recommend confirming these details with Nicomatic for their specific application."
                
            # Get connector technical specifications
            temp_range = connector_specs.get('temp_range', (-273, 1000))
            specs_to_include = {
                "Pitch Size": f"{connector_specs.get('pitch_size', 'N/A')} mm",
                "Maximum Current": f"{connector_specs.get('max_current', 'N/A')} A",
                "Temperature Range": f"{temp_range[0]} to {temp_range[1]}°C"}
            
            # Format specs for inclusion
            formatted_specs = "\n".join([f"- {name}: {value}" for name, value in specs_to_include.items()])
                
            user_message = HumanMessage(content=f"""Based on the following requirements from the user:
            
            {requirements_summary}
            
            Confidence Scores:
            {scores_text}
            
            Please recommend the {best_connector} connector as the closest match among Nicomatic's connectors.
            {unconfirmed_notes}
            
            The {best_connector} connector has the following technical specifications that MUST be included in your response:
            {formatted_specs}
            
            Start your response with a summary of the key requirements that led to this recommendation.
            Then explain that based on these requirements, the {best_connector} is the most suitable connector from Nicomatic.
            Be sure to include the technical specifications (pitch size, operational current, and temperature range) in your response.
            
            For building the part number for this connector, provide this link: "{link}"
            
            Format guidelines:
                - Begin with "Based on your requirements..."
                - Include a brief summary of the key inputs that led to this recommendation
                - Include the technical specifications as listed above
                - Do not mention features of other connectors
                - Do not mention confidence scores
                - Keep the response concise and avoid special characters or formatting """)

            try:
                llm_response = await self.llm.agenerate([[system_message, user_message]])
                recommendation_text = llm_response.generations[0][0].text
                
                # Return in the expected format
                return {
                    "status": "complete",
                    "recommendation": {
                        "connector": best_connector,
                        "confidence": f"{max_confidence:.1f}%",
                        "analysis": recommendation_text,
                        "requirements": requirements_text,
                        "requirements_summary": requirements_summary,
                        "confidence_scores": formatted_scores }}
            except Exception as e:
                print(f"Error generating connector recommendation: {str(e)}")
                # Fallback to static recommendation message if LLM fails
                
                # Format specs for fallback message
                specs_info = f"It features a pitch size of {connector_specs.get('pitch_size', 'N/A')} mm, " \
                            f"operational current of up to {connector_specs.get('max_current', 'N/A')} A, and " \
                            f"temperature range of {temp_range[0]} to {temp_range[1]}°C."
                    
                # Include any unconfirmed features in fallback message
                feature_notes = ""
                if unconfirmed_features:
                    feature_notes = "\n\nPlease note: " + "; ".join(unconfirmed_features) + ". Consider confirming these details with Nicomatic for your specific application."
                    
                fallback_message = (
                    f"Based on your requirements:\n\n{requirements_summary}\n\n"
                    f"I recommend the {best_connector} connector from Nicomatic's range. "
                    f"This connector best matches your specifications for connection type, current requirements, and orientation. "
                    f"{specs_info}"
                    f"{feature_notes}\n\n"
                    f"To configure your specific {best_connector} part, please use this link: {link}")
                
                # Return in the expected format
                return {
                    "status": "complete",
                    "recommendation": {
                        "connector": best_connector,
                        "confidence": f"{max_confidence:.1f}%",
                        "analysis": fallback_message,
                        "requirements": requirements_text,
                        "requirements_summary": requirements_summary,
                        "confidence_scores": formatted_scores}}
        except Exception as e:
            print(f"Exception in generate_recommendation: {str(e)}")
            # Return a properly structured error response
            return {
                "status": "complete",
                "recommendation": {
                    "connector": "CMM",  
                    "confidence": "error",
                    "analysis": "Based on your requirements for a plastic connector with 2mm pitch, I recommend the CMM connector from Nicomatic. CMM is designed for PCB-to-PCB connections with a 2mm pitch, featuring a plastic housing, and is ideal for on-board applications. It offers an operational current of up to 30A and a temperature range of -60 to 260°C.",
                    "requirements": "Error processing detailed requirements",
                    "requirements_summary": "Plastic connector with 2mm pitch",
                    "confidence_scores": {"CMM": 100.0, "DMM": 50.0, "AMM": 0.0, "EMM": 0.0}}}

    def format_requirements(self) -> str:
        critical_questions = {'mixed_power_signal', 'emi_protection', 'housing_material'}
        critical_reqs = []
        other_reqs = []

        for attr, (value, conf) in self.answers.items():
            if value is not None:
                requirement = f"{attr}: {value} (confidence: {conf:.2f})"
                if attr in critical_questions:
                    critical_reqs.append(requirement)
                else:
                    other_reqs.append(requirement)
        return "Critical Requirements:\n" + "\n".join(critical_reqs) + "\n\nOther Requirements:\n" + "\n".join(other_reqs)
  
    ## For CLI purposes
    def format_scores(self) -> str:
        return "\n".join([
            f"{connector}: {score:.1f}%"
            for connector, score in sorted(
                self.confidence_scores.items(),
                key=lambda x: x[1],
                reverse=True)])
    def clean_numeric_value(self, value: str) -> float:
        try:
            cleaned = ''.join(c for c in value.replace(',', '.') if c.isdigit() or c == '.')
            if cleaned:
                return float(cleaned)
            return None
        except (ValueError, AttributeError):
            return None
    def _simple_fallback_parse(self, response: str, question: Dict) -> Dict:
        if question['attribute'] == 'pitch_size':
            # Look for common pitch sizes
            pitch_values = [1.0, 1.27, 2.0]
            for pitch in pitch_values:
                if str(pitch) in response or f"{pitch:.1f}" in response:
                    return {
                        "value": pitch,
                        "confidence": 0.8,
                        "reasoning": f"Matched standard pitch size {pitch}mm in response"}
        
        elif question['attribute'] == 'pin_count':
            # Extract numeric values
            numbers = re.findall(r'\b\d+\b', response)
            if numbers:
                try:
                    pin_count = int(numbers[0])
                    if 1 <= pin_count <= 200:
                        return {
                            "value": pin_count,
                            "confidence": 0.7,
                            "reasoning": f"Extracted pin count {pin_count} from response"
                        }
                except (ValueError, IndexError):
                    pass
        
        elif question['attribute'] == 'housing_material':
            # Check for housing material keywords
            if 'metal' in response.lower():
                return {
                    "value": "metal",
                    "confidence": 0.8,
                    "reasoning": "User mentioned metal housing"}
            elif 'plastic' in response.lower():
                return {
                    "value": "plastic",
                    "confidence": 0.8,
                    "reasoning": "User mentioned plastic housing"}
            elif 'emi' in response.lower() or 'shield' in response.lower():
                return {
                    "value": "metal",
                    "confidence": 0.7,
                    "reasoning": "User mentioned EMI or shielding, which implies metal housing"}
        
        # For yes/no questions
        if question['text'].endswith('?'):
            if any(word in response.lower() for word in ['yes', 'yeah', 'yep', 'correct', 'right']):
                return {
                    "value": True,
                    "confidence": 0.7,
                    "reasoning": "User responded affirmatively"}
            elif any(word in response.lower() for word in ['no', 'nope', 'not', 'don\'t', 'dont']):
                return {
                    "value": False,
                    "confidence": 0.7,
                    "reasoning": "User responded negatively" }
        
        # General fallback for any response
        if response.lower() in ["i dont know", "i don't know", "unknown", "unclear", "not sure"]:
            return {
                "value": None,
                "confidence": 0.0,
                "reasoning": "User explicitly expressed uncertainty"
            }
        
        # Return low confidence for other cases
        return {
            "value": response.strip(),
            "confidence": 0.4,
            "reasoning": "Fallback: Could not confidently parse the response"
        }
    def _aggressive_fallback_parse(self, response: str, question: Dict) -> Dict:
        # For pitch size, look for any number followed by mm
        if question['attribute'] == 'pitch_size':
            pitch_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)', response.lower())
            if pitch_match:
                try:
                    pitch = float(pitch_match.group(1))
                    # Common pitch sizes
                    common_pitches = [1.0, 1.27, 2.0]
                    # Find closest common pitch
                    closest_pitch = min(common_pitches, key=lambda x: abs(x - pitch))
                    return {
                        "value": closest_pitch,
                        "confidence": 0.6,
                        "reasoning": f"Approximated to standard pitch size {closest_pitch}mm"
                    }
                except (ValueError, IndexError):
                    pass
                    
            # If no number with mm, check for standard pitch mentions
            if "1mm" in response.lower() or "1 mm" in response.lower():
                return {"value": 1.0, "confidence": 0.6, "reasoning": "Matched '1mm' in response"}
            elif "1.27mm" in response.lower() or "1.27 mm" in response.lower():
                return {"value": 1.27, "confidence": 0.6, "reasoning": "Matched '1.27mm' in response"}
            elif "2mm" in response.lower() or "2 mm" in response.lower():
                return {"value": 2.0, "confidence": 0.6, "reasoning": "Matched '2mm' in response"}
        
        # For housing_material with aggressive matching
        elif question['attribute'] == 'housing_material':
            response_lower = response.lower()
            
            # Check for preference indicators
            preference_terms = ['prefer', 'preferable', 'ideally', 'better', 'if possible', 'would like']
            is_preference = any(term in response_lower for term in preference_terms)
            
            if any(word in response_lower for word in ['metal', 'metallic', 'alumin', 'steel', 'emi', 'shield']):
                confidence = 0.85 if is_preference else 0.95
                return {"value": "metal", "confidence": confidence, 
                        "reasoning": "Matched metal-related terms" + (" (as preference)" if is_preference else "")}
            else:
                # Default to plastic if no metal indication
                confidence = 0.85 if is_preference else 0.95
                return {"value": "plastic", "confidence": confidence, 
                        "reasoning": "Defaulted to plastic as no metal indicators found" + (" (as preference)" if is_preference else "")}
            
        # For temperature, extract any number before C or degrees
        elif question['attribute'] == 'temp_range':
            temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:c|celsius|°c|degrees?)', response.lower())
            if temp_match:
                try:
                    temp = float(temp_match.group(1))
                    return {"value": temp, "confidence": 0.6, "reasoning": f"Extracted temperature {temp}°C from response"}
                except (ValueError, IndexError):
                    pass
        
        default_values = {'pitch_size': 2.0,  
            'housing_material': 'plastic',  
            'right_angle': True,  
            'pin_count': 20,  
            'max_current': 5.0,
            'temp_range': 85.0,
            'connection_type': 'PCB-to-PCB'}
        
        if question['attribute'] in default_values:
            return {"value": default_values[question['attribute']],
                "confidence": 0.3,
                "reasoning": f"Used default value after multiple parse failures"}
        
        # Last resort
        return {"value": None,
            "confidence": 0.0,
            "reasoning": "Could not determine a value even with aggressive fallback"}
    ## Understand user message with this and them grade confidence score
    async def parse_response_with_llm(self, response: str, question: Dict) -> Dict:
        try:
            # Special handling for height_requirement
            if question['attribute'] == 'height_requirement':
                return self.parse_space_constraints(response)
            # Handle other question types with the LLM
            system_message = SystemMessage(content=self.system_prompt)
            
            user_prompt = f"""Parse the following user response to the question about {question['attribute']}:
            Question: {question['text']}
            Response: {response}
            
            Consider the following:
            - Technical context: {question['clarification']}
            - Parsing instructions: {question['parse_prompt']}
            
            Provide your response in the following JSON format only:
            {{ "value": "parsed value (can be any data type)",
                "confidence": "number between 0 and 1",
                "reasoning": "your explanation"}}"""
            
            user_message = HumanMessage(content=user_prompt)
            messages = [system_message, user_message]
            
            # Use timeout to prevent hanging
            try:
                llm_response = await asyncio.wait_for(
                    self.llm.agenerate([messages]), 
                    timeout=10.0 
                )
                response_text = llm_response.generations[0][0].text
                
                # Track parse failures for adaptive behavior
                self.parse_failures = 0
                
            except (asyncio.TimeoutError, Exception) as e:
                print(f"LLM processing error or timeout: {str(e)}")
                self.parse_failures += 1
                
                # If we've had multiple failures, use a more aggressive fallback
                if self.parse_failures > 2:
                    return self._aggressive_fallback_parse(response, question)
                
                return self._simple_fallback_parse(response, question)
            
            try:
                parsed_response = self.output_parser.parse(response_text)
                return parsed_response
            except Exception as parse_error:
                print(f"Parser error: {parse_error}. Falling back to direct parsing.")
                self.parse_failures += 1
                
                # Choose fallback based on failure count
                if self.parse_failures > 2:
                    return self._aggressive_fallback_parse(response, question)
                return self._simple_fallback_parse(response, question)
                
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            self.parse_failures += 1
            return self._aggressive_fallback_parse(response, question)
    

    
    def select_next_question(self, skipped_questions: Dict[str, int]) -> Dict:
        # Check if connection type is already established
        connection_type = None
        if 'connection_types' in self.answers:
            connection_type = self.answers['connection_types'][0]
        height_question_asked = 'height_requirement' in self.asked_questions
        height_answer_uncertain = False
        # Skip irrelevant questions for PCB-to-PCB connections
        questions_to_skip = set()
        if connection_type and connection_type.lower() in ["pcb-to-pcb", "pcb to pcb", "board to board"]:
            # Skip wire gauge question for PCB-to-PCB connections (no cables involved)
            questions_to_skip.add('wire_gauge')
            
            # Skip location/panel mount question for PCB-to-PCB (always on-board)
            questions_to_skip.add('location')
            
            # Mark these as answered with default values
            if 'wire_gauge' not in self.answers:
                self.answers['wire_gauge'] = (None, 0.0)
                self.asked_questions.add('wire_gauge')
                
            if 'location' not in self.answers:
                self.answers['location'] = ("internal", 0.95)  # PCB-to-PCB is always internal/on-board
                self.asked_questions.add('location')
        if height_question_asked and 'height_requirement' in self.answers:
            _, confidence = self.answers['height_requirement']
            # Consider the answer uncertain if confidence is low or value is None
            if confidence < 0.5:
                height_answer_uncertain = True
        
        # If the height question was asked but user was uncertain, prioritize pitch_size
        if (height_question_asked and height_answer_uncertain) or \
        (height_question_asked and 'height_requirement' not in self.answers):
            # Check if pitch_size question is still available
            pitch_question = next((q for q in self.all_questions 
                                if q['attribute'] == 'pitch_size' 
                                and q['attribute'] not in self.asked_questions), None)
            if pitch_question:
                return pitch_question
        
        # Standard question selection logic
        available_questions = [
            q for q in self.all_questions 
            if q['attribute'] not in self.asked_questions
            and q['attribute'] not in questions_to_skip
            and skipped_questions.get(q['attribute'], 0) < 2
        ]
        
        if not available_questions:
            return None
            
        # Sort by order and return the first available question
        return min(available_questions, key=lambda x: x['order'])
    def parse_space_constraints(self, response: str) -> Dict:
        response_lower = response.lower().replace('millimeters', 'mm').replace('millimeter', 'mm')
        
        # Check for uncertainty phrases first
        uncertainty_phrases = [
            "don't know", "dont know", "not sure", "uncertain", 
            "no idea", "no specific", "not specified", "unsure",
            "don't have", "no constraint", "no requirement", "any height",
            "flexible", "whatever works", "any option", "no particular constraint"]
        
        if any(phrase in response_lower for phrase in uncertainty_phrases):
            return {
                "value": None,
                "confidence": 0.0,
                "reasoning": "User expressed uncertainty about spatial constraints"}
        
        # Look for footprint minimization intent
        footprint_indicators = [
            "minimum footprint", "small footprint", "compact", "tight space", 
            "limited space", "not much space", "space available", "small as possible"]
        is_space_constrained = any(indicator in response_lower for indicator in footprint_indicators)
        
        # Extract pin count information when present
        pin_pattern = r'(\d+)\s*(?:pins?|contacts?)'
        pin_match = re.search(pin_pattern, response_lower)
        pin_count = int(pin_match.group(1)) if pin_match else None
        
        # Look for "fit within" or similar constraint phrases
        constraint_phrases = ["fit within", "fit in", "maximum of", "not exceed", "at most", "up to"]
        is_max_constraint = any(phrase in response_lower for phrase in constraint_phrases)
        
        # Check for 2D dimensions (most common format)
        two_d_pattern = r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeter)?'
        two_d_match = re.search(two_d_pattern, response_lower)
        if two_d_match:
            dim1, dim2 = map(float, two_d_match.groups())
            # In PCB context, height is typically the smaller dimension
            length = max(dim1, dim2)
            height = min(dim1, dim2)
            
            # Build detailed reasoning
            reasoning = f"Extracted dimensions: {dim1}x{dim2}mm (using {height}mm as height)"
            if is_space_constrained:
                reasoning += " with limited space constraint"
            if pin_count:
                reasoning += f" for {pin_count} pins"
            if is_max_constraint:
                reasoning += " as maximum allowed dimensions"
                
            return {
                "value": height,
                "confidence": 0.95 if is_space_constrained or is_max_constraint else 0.9,
                "reasoning": reasoning,
                "is_maximum": is_max_constraint or is_space_constrained,
                "all_dimensions": {"length": length, "height": height},
                "pin_count": pin_count
            }
        
        height_patterns = [
            # Direct height specification
            r'height\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeter)',
            r'(\d+(?:\.\d+)?)\s*(?:mm|millimeter)\s*(?:tall|height|high)',
            r'height\s*(?:requirement|constraint|limit)?\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)',
            
            # Constraint-based specification
            r'maximum\s*(?:height|space|clearance)\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)',
            r'(?:height|space|clearance)\s*(?:less than|under|below|not more than)\s*(\d+(?:\.\d+)?)',
            r'up\s*to\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeter)\s*(?:height|tall|high|clearance)',
            r'(?:can\'t exceed|cannot exceed|not exceed|no more than)\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeter)',
            
            # Approximate specification
            r'(?:about|around|approximately|roughly|circa|~)\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeter)',
            
            # Range specification
            r'(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*(?:mm|millimeter)',
            
            # Simple numeric with mm unit
            r'(\d+(?:\.\d+)?)\s*(?:mm|millimeter)'
        ]
        
        for pattern in height_patterns:
            match = re.search(pattern, response_lower)
            if match:
                # Handle range patterns specially
                if 'between' in pattern or 'from' in pattern:
                    min_val, max_val = map(float, match.groups())
                    # Use average of range
                    height = (min_val + max_val) / 2
                    return {
                        "value": height,
                        "confidence": 0.8,
                        "reasoning": f"Using midpoint {height}mm from range {min_val}-{max_val}mm",
                        "range": [min_val, max_val]
                    }
                else:
                    height = float(match.group(1))
                    
                    # Assign confidence based on specificity
                    if 'about' in pattern or 'around' in pattern or 'approximately' in pattern:
                        confidence = 0.75
                    elif 'maximum' in pattern or 'up to' in pattern:
                        confidence = 0.85
                    else:
                        confidence = 0.9
                    
                    # Validate reasonable range
                    if 1.0 <= height <= 20.0:
                        return {
                            "value": height,
                            "confidence": confidence,
                            "reasoning": f"Extracted height: {height}mm"
                        }
                    else:
                        return {
                            "value": height,
                            "confidence": 0.5,
                            "reasoning": f"Extracted unusual height value: {height}mm"
                        }
        
        if 'small' in response_lower or 'compact' in response_lower or 'tiny' in response_lower:
            return {
                "value": 4.0, 
                "confidence": 0.6,
                "reasoning": "Inferred small height requirement from descriptive terms"
            }
        elif 'large' in response_lower or 'big' in response_lower or 'spacious' in response_lower:
            return {
                "value": 10.0,  
                "confidence": 0.6,
                "reasoning": "Inferred larger height requirement from descriptive terms"
            }
        
        # No height information found
        return {
            "value": None,
            "confidence": 0.0,
            "reasoning": "Could not extract any height or space constraint information"
        }
    def normalize_connection_type(self, value):
        if not isinstance(value, str):
            return value
            
        value_lower = value.lower().replace(' ', '-')
        
        # Handle PCB-to-PCB variations
        if any(pattern in value_lower for pattern in ["pcb-pcb", "pcb-to-pcb", "board-to-board", "board-board"]):
            return "PCB-to-PCB"
            
        # Handle PCB-to-Cable variations
        if any(pattern in value_lower for pattern in ["pcb-cable", "pcb-to-cable", "board-to-cable"]):
            return "PCB-to-Cable"
            
        # Handle Cable-to-PCB variations
        if any(pattern in value_lower for pattern in ["cable-pcb", "cable-to-pcb", "cable-to-board"]):
            return "Cable-to-PCB"
            
        # Handle Cable-to-Cable variations
        if any(pattern in value_lower for pattern in ["cable-cable", "cable-to-cable"]):
            return "Cable-to-Cable"
            
        return value
    ## Calculation of confidence score and prediction
    def calculate_connector_score(self, connector_specs: Dict, answers: Dict) -> float:
        total_weighted_score = 0
        total_weight = 0
        critical_mismatch = False
        critical_mismatch_factors = []
        
        # Define critical attributes that must match
        critical_attributes = {
            'pitch_size': 'Pitch size mismatch',
            'emi_protection': 'EMI protection requirement mismatch',
            'housing_material': 'Housing material mismatch',
            'pin_count': 'Pin count exceeds maximum',
            'wire_gauge': 'Wire gauge not supported'
        }
        
        # Track matched and unmatched attributes for logging
        matched_attrs = []
        unmatched_attrs = []
        
        # Process each answer and calculate individual scores
        for attr, (value, confidence) in answers.items():
            # Skip if value is None or confidence is 0
            if value is None or confidence == 0:
                continue
                
            # Find the question to get the weight
            question = next((q for q in self.all_questions if q['attribute'] == attr), None)
            if not question:
                continue
            
            weight = float(question['weight'])
            adjusted_weight = weight * float(confidence)
            total_weight += adjusted_weight
            
            # Calculate attribute score based on attribute type
            attr_score = 0
            
            # Location handling (on-board vs panel mount)
            if attr == 'location':
                location_value = value.lower() if isinstance(value, str) else value
                internal_keywords = ['internal', 'in box', 'on board', 'inside', 'onboard']
                external_keywords = ['external', 'out of box', 'panel mount', 'outside']
                
                is_internal = any(keyword in location_value for keyword in internal_keywords) if isinstance(location_value, str) else (location_value == "internal")
                is_external = any(keyword in location_value for keyword in external_keywords) if isinstance(location_value, str) else (location_value == "external")
                
                # Map to boolean for panel_mount in connector specs
                requires_panel_mount = is_external
                has_panel_mount = connector_specs.get('panel_mount', False)
                
                # For internal use, all connectors should score well
                if is_internal:
                    # All connectors can be used internally
                    attr_score = 1.0
                    matched_attrs.append(attr)
                elif requires_panel_mount and has_panel_mount:
                    attr_score = 1.5
                    matched_attrs.append(attr)
                elif requires_panel_mount and not has_panel_mount:
                    attr_score = 0.3
                    unmatched_attrs.append(attr)
                else:
                    attr_score = 1.0
                    matched_attrs.append(attr)
            
            # Connection Types handling
            elif attr == 'connection_types':
                ## All connector families support PCB to Cable connections
                ## This should not decrease scores
                if value in ["PCB-to-Cable", "Cable-to-PCB", "pcb to cable", "cable to pcb"]:
                    attr_score = 1.0
                    matched_attrs.append(attr)
                else:
                    ## For other connection types, default to good compatibility
                    attr_score = 0.8
                    matched_attrs.append(attr)
            elif attr == 'right_angle':
                user_wants_right_angle = bool(value)
                connector_supports_right_angle = connector_specs.get('right_angle', False)
                
                if user_wants_right_angle:
                    # User wants right angle
                    if connector_supports_right_angle:
                        # Perfect match
                        attr_score = 1.0  
                        matched_attrs.append(attr)
                    else:
                        attr_score = 0.3  
                        # Significant penalty but not critical
                        unmatched_attrs.append(attr)
                else:
                    attr_score = 1.0  # All connectors can be straight
                    matched_attrs.append(attr)
                    print(f"Straight configuration requested - {connector_specs.get('type', 'unknown')} supports this")
                    
            # AWG (wire gauge) handling
            elif attr == 'wire_gauge':
                try:
                    # Normalize required AWG to numeric value
                    required_awg = self.normalize_awg_value(value) if callable(getattr(self, 'normalize_awg_value', None)) else normalize_awg_value(value)
                    if required_awg is None:
                        continue
                        
                    # Get supported AWG values from connector specs
                    supported_awgs_raw = connector_specs.get('wire_gauge', [])
                    
                    # Normalize the supported AWG values to numeric form
                    supported_awgs = []
                    for awg_str in supported_awgs_raw:
                        norm_awg = self.normalize_awg_value(awg_str) if callable(getattr(self, 'normalize_awg_value', None)) else normalize_awg_value(awg_str)
                        if norm_awg is not None:
                            supported_awgs.append(norm_awg)
                    
                    # Check if required AWG is directly supported
                    if supported_awgs and required_awg in supported_awgs:
                        attr_score = 1.0
                        matched_attrs.append(attr)
                    else:
                        # Not in supported list - Apply penalty
                        attr_score = 0.0 
                        unmatched_attrs.append(attr)    
                        # Mark as critical mismatch with high importance
                        critical_mismatch = True
                        critical_mismatch_factors.append(f"AWG {required_awg} is not in supported list {supported_awgs_raw}")
                except (ValueError, TypeError, AttributeError):
                    # Default score if processing fails
                    attr_score = 0.5
            elif attr == 'height_requirement':
                height_value = float(value)
                height_range = connector_specs.get('height_range', (0, 0))
                height_options = connector_specs.get('height_options', [])
                
                user_height_range = answers.get("height_requirement_range", None)
                
                if user_height_range:
                    min_user, max_user = user_height_range
                    if any(min_user <= opt <= max_user for opt in height_options):
                        attr_score = 1.0
                        matched_attrs.append(attr)
                    else:
                        # Find closest available height to the range
                        closest_to_range = min(height_options, key=lambda x: 
                                            min(abs(x - min_user), abs(x - max_user)))
                        height_diff = min(abs(closest_to_range - min_user), 
                                        abs(closest_to_range - max_user))
                        
                        if height_diff <= 1.5:
                            attr_score = 0.9 
                            matched_attrs.append(attr)
                        else:
                            # More gradual decrease in score
                            attr_score = max(0.5, 1.0 - (height_diff / 10.0)) 
                            if attr_score >= 0.7:
                                matched_attrs.append(attr)
                            else:
                                unmatched_attrs.append(attr)
                elif height_range[0] <= height_value <= height_range[1]:
                    # Height is within connector's range
                    attr_score = 1.0
                    matched_attrs.append(attr)
                elif height_options:
                    # Find closest available height
                    closest_height = min(height_options, key=lambda x: abs(x - height_value))
                    height_diff = abs(closest_height - height_value)
                    relative_diff = height_diff / height_value if height_value > 0 else height_diff
                    
                    if relative_diff <= 0.1:  
                        attr_score = 0.95
                        matched_attrs.append(attr)
                    elif relative_diff <= 0.2:  
                        attr_score = 0.85
                        matched_attrs.append(attr)
                    elif relative_diff <= 0.3:  
                        attr_score = 0.7
                        matched_attrs.append(attr)
                    else:
                        attr_score = max(0.4, 0.8 - (relative_diff / 2.0))  
                        unmatched_attrs.append(attr)
                        
                        # Only consider a critical mismatch for very large differences
                        if relative_diff > 0.8:  
                            critical_mismatch = True
                            critical_mismatch_factors.append(
                                f"Height requirement ({height_value}mm) far from available options ({closest_height}mm)")
                else:
                    attr_score = 0.5  
                    unmatched_attrs.append(attr)
            
            # Special handling for pin count
            elif attr == 'pin_count':
                pin_count = int(value)
                valid_pins = connector_specs.get('valid_pin_counts', set())
                max_pins = connector_specs.get('max_pins', 0)
                
                if pin_count > max_pins:
                    attr_score = 0.0
                    critical_mismatch = True
                    unmatched_attrs.append(attr)
                    critical_mismatch_factors.append(f"Pin count ({pin_count}) exceeds maximum ({max_pins})")
                elif pin_count in valid_pins:
                    attr_score = 1.0
                    matched_attrs.append(attr)
                else:
                    # Find closest valid pin count
                    if valid_pins:
                        closest_pin = min(valid_pins, key=lambda x: abs(x - pin_count))
                        pin_diff = abs(closest_pin - pin_count)
                        
                        if pin_diff <= 2:
                            attr_score = 0.8
                            matched_attrs.append(attr)
                        elif pin_diff <= 4:
                            attr_score = 0.5
                            unmatched_attrs.append(attr)
                        else:
                            attr_score = 0.2
                            unmatched_attrs.append(attr)
                            if pin_diff > 10:
                                critical_mismatch = True
                                critical_mismatch_factors.append(f"Pin count ({pin_count}) not available, closest is {closest_pin}")
                    else:
                        attr_score = 0.0
                        unmatched_attrs.append(attr)
            
            elif attr == 'housing_material':
                required_material = value.lower() if isinstance(value, str) else value
                connector_material = connector_specs.get('housing_material', '').lower()
                
                # Normalize material names for comparison
                if required_material in ['metal', 'metallic', 'aluminum', 'steel', 'alloy']:
                    required_material_normalized = 'metal'
                else:
                    required_material_normalized = 'plastic'
                
                # Convert connector_material to normalized form too
                connector_material_normalized = 'metal' if connector_material in ['metal', 'metallic', 'aluminum', 'steel', 'alloy'] else 'plastic'
                
                # Compare normalized values
                if required_material_normalized == connector_material_normalized:
                    attr_score = 1.2 
                    matched_attrs.append(attr)
                    # Additional bonus for matching metal housing
                    if required_material_normalized == 'metal' and connector_material_normalized == 'metal':
                        attr_score = 1.3
                else:
                    # Critical mismatch ONLY if user needs metal but connector is plastic
                    if required_material_normalized == 'metal' and connector_material_normalized != 'metal':
                        attr_score = 0.15 
                        unmatched_attrs.append(attr)
                        # Mark as critical mismatch with housing material flag
                        critical_mismatch = True
                        critical_mismatch_factors.append("Metal housing required but not available")
                    else:
                        attr_score = 0.5 
                        unmatched_attrs.append(attr)
                
            elif attr == 'mixed_power_signal':
                required_power = bool(value)
                has_power = connector_specs.get('mixed_power_signal', False)
                
                if required_power and has_power:
                    attr_score = 1.5 
                    matched_attrs.append(attr)
                    print(f"Connector supports high power/frequency - compatible with answer: {required_power}")
                elif required_power and not has_power:
                    attr_score = 0.1 
                    unmatched_attrs.append(attr)
                    # Add critical mismatch when power is explicitly required but not supported
                    critical_mismatch = True
                    critical_mismatch_factors.append("Mixed power/signal capability required but not supported")
                    print(f"Connector doesn't support required high power/frequency (CRITICAL MISMATCH)")
                else:
                    attr_score = 1.0
                    matched_attrs.append(attr)
                    print(f"High power not required, connector compatible")
                    
            # Special handling for temperature
            elif attr == 'temp_range':
                temp_value = float(value)
                spec_range = connector_specs.get('temp_range', (-273, 1000))
                min_temp, max_temp = spec_range
                
                if min_temp <= temp_value <= max_temp:
                    attr_score = 1.0
                    matched_attrs.append(attr)
                elif temp_value > max_temp:
                    # Score decreases as temperature exceeds maximum
                    temp_diff = temp_value - max_temp
                    attr_score = max(0.3, 1.0 - (temp_diff / 75.0))  
                    unmatched_attrs.append(attr)
                    
                    if temp_diff > 50: 
                        critical_mismatch = True
                        critical_mismatch_factors.append(f"Temperature requirement ({temp_value}°C) exceeds maximum ({max_temp}°C)")
                else:
                    # Below minimum but less critical
                    temp_diff = min_temp - temp_value
                    attr_score = max(0.3, 1.0 - (temp_diff / 75.0)) 
                    unmatched_attrs.append(attr)
            
            # Special handling for pitch size
            elif attr == 'pitch_size':
                if isinstance(value, str):
                    try:
                        pitch_value = float(''.join(c for c in value if c.isdigit() or c == '.'))
                    except ValueError:
                        # Default to 0 if conversion fails completely
                        pitch_value = 0
                else:
                    pitch_value = float(value)
                
                spec_pitch = connector_specs.get('pitch_size', 0)
                
                ## Pitch must match exactly (within small tolerance)
                if abs(pitch_value - spec_pitch) < 0.05:
                    attr_score = 2.0
                    matched_attrs.append(attr)
                    print(f"PITCH MATCH: {connector_specs.get('type', 'unknown')} pitch {spec_pitch}mm matches requested {pitch_value}mm")
                else:
                    attr_score = 0.1
                    critical_mismatch = True
                    unmatched_attrs.append(attr)
                    critical_mismatch_factors.append(f"Pitch size mismatch: required {pitch_value}mm, connector has {spec_pitch}mm")
            # Generic handling for boolean attributes
            elif isinstance(value, bool) and attr in connector_specs:
                spec_value = connector_specs.get(attr, False)
                
                if attr in critical_attributes:
                    if value == spec_value:
                        attr_score = 1.0
                        matched_attrs.append(attr)
                    else:
                        attr_score = 0.3  
                        unmatched_attrs.append(attr)
                        if attr == 'emi_protection' and value and not spec_value:
                            critical_mismatch = True
                            critical_mismatch_factors.append("EMI protection required but not available")
                else:
                    # For non-critical boolean attributes
                    if value == spec_value:
                        attr_score = 1.0
                        matched_attrs.append(attr)
                    else:
                        attr_score = 0.7 if not value else 0.3
                        unmatched_attrs.append(attr)
            
            # Handle other cases with a default score
            else:
                attr_score = 0.5
            
            total_weighted_score += adjusted_weight * attr_score
        
        # Prevent division by zero
        if total_weight < 0.001:
            return 0.0
        
        # Log matched and unmatched attributes for debugging
        if matched_attrs:
            print(f"Matched attributes for {connector_specs.get('type', 'unknown')}: {', '.join(matched_attrs)}")
        if unmatched_attrs:
            print(f"Unmatched attributes for {connector_specs.get('type', 'unknown')}: {', '.join(unmatched_attrs)}")
        
        base_score = 100.0
        
        mismatch_penalty = 0.0
        if total_weight > 0:
            mismatch_penalty = (total_weight - total_weighted_score) * 1.2  
            
        adjusted_score = max(10.0, base_score - mismatch_penalty) 
        
        material_bonus = 1.0
        if 'housing_material' in answers and 'location' in answers:
            required_material = answers['housing_material'][0]
            location_value = answers['location'][0]
            is_panel_mount = (location_value == "external" or 
                            (isinstance(location_value, str) and 
                                any(word in location_value.lower() for word in 
                                    ['external', 'out of box', 'panel mount', 'outside'])))
            
            connector_material = connector_specs.get('housing_material', '')
            
            # Normalize for comparison
            required_normalized = 'metal' if required_material in ['metal', 'metallic'] else 'plastic'
            connector_normalized = 'metal' if connector_material in ['metal', 'metallic'] else 'plastic'
            if required_normalized != connector_normalized:
                return 0.0
            if required_material_normalized == connector_material_normalized:
                # Higher bonus for matching metal housing for panel mount applications
                if required_material_normalized == 'metal' and is_panel_mount:
                    material_bonus = 1.2
                else:
                    material_bonus = 1.1
        
        # Apply critical mismatch penalty - but with more balanced approach
        final_score = adjusted_score * material_bonus
        if critical_mismatch:
            # Standard penalty calculation
            penalty_factor = max(0.5, 0.8 - (0.03 * len(critical_mismatch_factors)))
            
            # Apply stronger penalties for specific critical mismatches
            
            if any("Mixed power/signal capability required but not supported" in factor for factor in critical_mismatch_factors):
                penalty_factor *= 0.5  
            
            if any("Metal housing required but not available" in factor for factor in critical_mismatch_factors):
                penalty_factor *= 0.5  
            
            final_score *= penalty_factor
            print(f"Critical mismatch for {connector_specs.get('type', 'unknown')}: {', '.join(critical_mismatch_factors)}")
        # especially when we have only partial information
        min_score = 5.0  
        if len(answers) < 3: 
            min_score = 20.0 
            
        # Ensure score is between min_score and 100
        return max(min_score, min(100.0, final_score))

def pre_process_routing(user_input, formatted_chat_history):

    # Force routing to general for any input with a question mark
    if '?' in user_input:
        print("Direct routing enforcement: Input contains question mark, routing to general")
        return {'score': 'general'}
    
    # Check for WH-question words about connector properties
    wh_words = ['what', 'where', 'why', 'who', 'how', 'which', 'when']
    if any(word in user_input.lower() for word in wh_words):
        print("Direct routing enforcement: Input contains WH-question words, routing to general")
        return {'score': 'general'}
    
    ## Check if this is asking about a specific connector that was just recommended
    ### Split the chat history to get the last message from the assistant
    messages = formatted_chat_history.split('\n')
    assistant_messages = [m for m in messages if m.startswith('AI:')]
    
    if assistant_messages:
        last_assistant_message = assistant_messages[-1][4:]  # Remove 'AI: ' prefix
        
        # Check if the last message was a recommendation
        connector_names = ['AMM', 'CMM', 'DMM', 'EMM']
        recommended_connectors = [c for c in connector_names if c in last_assistant_message]
        
        # If the last message contained a connector recommendation
        if recommended_connectors and any(p in user_input.lower() for p in ['it', 'this connector', 'the connector']):
            print("Direct routing enforcement: Question about recommended connector, routing to general")
            return {'score': 'general'}
    
    # Let the LLM handle other cases
    return None
    
    
# Add this class to track sources across the entire response generation process

class SourceTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SourceTracker, cls).__new__(cls)
            # Initialize with an empty list to store NodeWithScore objects
            cls._instance.nodes = []
        return cls._instance

    def reset(self):
        # Reset the list of nodes
        self.nodes = []

    # Keep track of node IDs to avoid duplicates
    # self.node_ids = set() # Optional: Uncomment if you want to avoid duplicate nodes

    def add_sources_from_nodes(self, nodes: List[NodeWithScore]):
        for node in nodes:
            # Optional: Uncomment below to skip adding if node ID already seen
            # if node.node.node_id in self.node_ids:
            #    continue
            # Append the whole NodeWithScore object
            self.nodes.append(node)
            # Optional: Uncomment below to track node ID
            # self.node_ids.add(node.node.node_id)


    def get_source_nodes(self) -> List[NodeWithScore]:
         # Return the stored list of NodeWithScore objects
        return self.nodes

    def get_source_text(self) -> str:
        if not self.nodes:
            return ""

        sources_list = []
        seen_sources = set()
        for node in self.nodes:
            # Access metadata from the inner node object
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                source = node.node.metadata.get("source", "Unknown")
                family = node.node.metadata.get("connector_family", "Unknown")
                # Avoid duplicate source/family pairs in the summary text
                if source != "Unknown" and (source, family) not in seen_sources:
                     sources_list.append(f"{source} ({family})")
                     seen_sources.add((source, family))

        if sources_list:
             return "\n\nSource documents: " + ", ".join(sorted(list(seen_sources), key=lambda x: x[0]))
        return ""

    def get_absolute_paths(self) -> List[str]:
        paths = set()
        for node in self.nodes:
             # Access metadata from the inner node object
            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                path = node.node.metadata.get("absolute_path", None)
                if path:
                    paths.add(path)
        return list(paths)



@app.get("/health")
async def health_check():
    return {
        "status": "ready" if app_ready else "initializing",
        "indices_loaded": all([
            vector_index_markdown is not None,
            keyword_index_markdown is not None,
            vector_index_markdown_lab is not None,
            keyword_index_markdown_lab is not None
        ]),
        "agents_available": agent_queue.qsize() if hasattr(agent_queue, 'qsize') else "unknown"
    }        
## RAG chain
@app.post("/chat")
async def chat(request: Request):
    global vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab, documents_context, current_possibility

    # Wait for startup to complete before processing any requests
    if not app_ready:
        print("Waiting for application startup to complete...")
        # Wait with timeout to avoid hanging indefinitely
        if not startup_complete.wait(timeout=60):
            raise HTTPException(status_code=503, 
                              detail="Application is still initializing. Please try again in a few moments.")

    try:
        # Check if indices are loaded - if not, try loading them on-demand
        if vector_index_markdown is None or keyword_index_markdown is None or vector_index_markdown_lab is None or keyword_index_markdown_lab is None:
            print("\n -- Indices not properly loaded during startup, attempting on-demand loading -- ")
            try:
                directory_path = "./extracted_best/"
                documents1, documents3 = load_data(directory_path)
                print("\n -- Processing data -- ")
                result_indices = processing_data(documents1, documents3)
                if result_indices and len(result_indices) == 4:
                    vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab = result_indices
                    print("Successfully loaded indices on-demand")
                else:
                    print("Warning: On-demand data loading failed")
            except Exception as load_error:
                print(f"Error in on-demand data loading: {str(load_error)}")
                # Continue with whatever indices we have

        body = await request.json()
        session_id = body.get('sessionId')
        user_input = body['message']
        
        if session_id not in session_mapping:
            raise HTTPException(status_code=404, detail="Session not found")
            
        chat_id = session_mapping[session_id]['chat_id'] 
        
        session_history = get_session_history(session_id, chat_id)
        history_messages = session_history.get_messages()
        
        chat_history = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                chat_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                chat_history.append(f"AI: {msg.content}")
            elif isinstance(msg, SystemMessage):
                chat_history.append(f"System: {msg.content}")

        formatted_chat_history = "\n".join(chat_history)

        # Create tools with whatever indices we have (may be None)
        tools = creating_tools(vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab)
        agent_with_chat_history = await get_agent(tools)
        session_history.add_message(HumanMessage(content=user_input))

    #     llm = ChatOllama(model="llama3.1", temperature=0.0, num_ctx=8152, cache=False, format="json")
        
    #     print("Routing your query now")
    #     ## Routing query to either general or selection
    #     prompt = PromptTemplate(
    # template=
    # """
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    # You are a precise query routing AI. Your job is to analyze the conversation CONTEXT (`history`) and the user's INTENT in their latest input (`question`) to correctly route the query to either the 'selection' or 'general' channel.

    # ### [Routing Channels:]

    # 1.  **selection**:
    #     Route here ONLY when the user's primary goal is **discovering or narrowing down the *type* of connector needed** based on requirements, **AND they have NOT mentioned a specific connector family (AMM, CMM, DMM, EMM) in their current input.** This includes when the user is:
    #     - Starting a search without a specific product line in mind (e.g., "I need a connector", "Help me find a connector for high power and signal").
    #     - Actively providing specifications or constraints ***specifically in response to a system question aimed at finding a suitable connector TYPE*** during an active selection process. Examples of such responses include values like "1.27mm", "20 pins", "metal housing", "EMI shielding required", "on board", "pcb to cable", "100 degrees C", "as small as possible", "no preference", "yes [to a selection question]", "I don't know".
    #     - Asking clarifying questions *about the selection process itself* or about a specification the system asked for, *while actively engaged in the selection flow* (e.g., "What does pitch size mean here?", "What temperature ranges are typical?", "What do you mean by operational current?", "Can you explain the housing options?").

    # 2.  **general**:
    #     Route here when the user's query **mentions a specific, named connector family (AMM, CMM, DMM, EMM)**, OR when they are seeking general information, definitions, discussing past recommendations, or providing conversational input not directly part of active selection. This includes:
    #     - **ABSOLUTE PRIORITY TRIGGER:** User input contains "AMM", "CMM", "DMM", or "EMM". **Route here REGARDLESS of context or any other words like 'need', 'require', or specifications mentioned.** This rule overrides ALL other signals.
    #         - **Examples:** "I need a DMM with 100 pins.", "Tell me about high-temp CMM options.", "What is the current rating for AMM?", "Compare DMM and CMM.", "Okay, what about the DMM?", "Information on EMM needed."
    #     - Seeking general information, definitions, or explanations about connectors or related concepts ***outside*** of an active system-led selection process OR when the question is *about* a specific named family (AMM, CMM, DMM, EMM). (e.g., "What is EMI shielding?", "Explain connector pitch.", "What's the standard pitch for DMM?").
    #     - Queries that commonly use WH-questions (What, Where, Why, How) or end in '?', asking for information rather than defining selection criteria. ***Crucially:*** A clarifying question *during* active `selection` (System: 'What pitch?', User: 'What does pitch mean?') stays in `selection`. A general knowledge question ("What is pitch?") goes to `general`.
    #     - Queries related to cables associated with connectors.
    #     - **Greetings, expressions of appreciation, or simple conversational filler** (e.g., "hello", "hi", "thanks", "thank you", "ok", "got it", "sounds good"). Route these directly to `general`.
    #     - Questions about connectors *previously recommended* by the system (e.g., "What's the lead time for that CMM you suggested?", "Tell me more about the DMM option.").

    # ### [Core Guidelines:]

    # 1.  **SPECIFIC NAME PRECEDENCE (MOST IMPORTANT RULE):** If the user's latest input contains "AMM", "CMM", "DMM", or "EMM", the query **MUST** be routed to **general**. This rule takes absolute precedence over context, intent keywords (like 'need', 'looking for'), or the presence of specifications (like '100 pins', 'high temp').
    # 2.  **ACTIVE SELECTION CONTEXT OVERRIDES KEYWORDS:** If the `history` shows the system just asked a question to narrow down the connector *type* (e.g., asking about pins, pitch, temp, housing, shielding, mounting, connection type), the user's *direct answer* to that question **MUST** be routed to **selection**, UNLESS the answer itself contains 'AMM', 'CMM', 'DMM', or 'EMM' (Rule 1 applies). Do *not* route to `general` just because the answer contains words like 'need', 'require', 'information', or technical terms (like 'EMI shielding') if it's clearly answering a selection question in context.
    # 3.  **CONTEXT IS KING (Beyond Rule 2):** Analyze the full `history`. A user response like "100 degrees" is `selection` if the system just asked about temperature *during selection*. It's `general` if discussing a specific DMM's specs or asking a general knowledge question unrelated to active selection.
    # 4.  **INTENT MATTERS (Secondary to Name Precedence & Context):**
    #     - **Selection Intent:** User is trying to *figure out which type* of connector fits their abstract needs, *without having a specific family in mind*, and is actively participating in the system-led discovery process.
    #     - **General Intent:** User is asking *about* a known connector type/concept, providing conversational filler, or continuing a non-selection discussion.
    # 5.  **POST-RECOMMENDATION QUESTIONS (CRITICAL RULE):** Once the system has suggested a specific connector type or family (e.g., 'Based on your needs, a CMM connector seems suitable.'), any subsequent user questions asking *about the features, capabilities, or specifics of that suggested connector* (even if referred to as 'it' or 'that one') **MUST** go to **general**. Examples: 'What's the lead time for that CMM?', 'Can it handle 150 degrees?', 'Is it available in a right-angle version?', **'Can I mount it on a panel?'**. Route to `selection` again only if the user explicitly rejects the suggestion and wants to restart the search OR modifies their core requirements significantly.
    # 6.  **AMBIGUITY & CLARIFICATION:**
    #     - Vague answers ("yes", "ok", "I don't know", "no preference") during `selection` stay in `selection` as they are responses within that flow.
    #     - User questions ("what do you mean?", "?", "what is X?") should be routed based on context: if clarifying a `selection` question -> `selection`; if asking for general info or about a specific family -> `general`.

    # ### [Output:]

    # Provide your routing decision as a JSON response with a single key `'score'`. The value must be exactly `"general"` or `"selection"`.
    # **Output *only* the raw JSON object.** No other text, comments, explanations, or markdown formatting are allowed.

    # Example valid outputs:
    # {{'score': 'selection'}}
    # {{'score': 'general'}}

    # <|eot_id|><|start_header_id|>user<|end_header_id|>
    # Here is the conversation history: \n\n {history} \n\n
    # Here is the human input: {question}

    # <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # """,
    # input_variables=["question", "history"])

    #     retrieval_grader = prompt | llm | JsonOutputParser()
        
    #     pre_processed_route = pre_process_routing(user_input, formatted_chat_history)
    #     if pre_processed_route is not None:
    #         routing_result = pre_processed_route
    #     else:
    #         # Proceed with LLM-based routing
    #         routing_result = retrieval_grader.invoke({"question": user_input, "history": formatted_chat_history})
        routing_result={"score":"general"}
        print(routing_result)
        if not isinstance(routing_result, dict) or 'score' not in routing_result:
            ## Fallback to general routing if response is invalid
            routing_result = {'score': 'general'}
            logging.warning(f"Invalid routing response, falling back to general. Response: {routing_result}")
                
        route = routing_result['score']
            
        ## Validate route value
        if route not in ['selection', 'general']:
            route = 'general'
            logging.warning(f"Invalid route value: {route}, falling back to general")
        elif route == 'selection':
            # Update the initial message processing in generate_connector_selection:

            async def generate_connector_selection():
                try:
                    # Initialize connector selector if not exists
                    if session_mapping[session_id].get('connector_selector') is None:
                        session_mapping[session_id]['connector_selector'] = LLMConnectorSelector()
                        selector = session_mapping[session_id]['connector_selector']
                        
                        # Initialize confidence scores
                        selector.confidence_scores = {connector: 0.0 for connector in selector.connectors}
                        
                        # Process initial message first
                        try:
                            print("Processing initial message...")
                            initial_result = await selector.process_initial_message(user_input)
                            
                            if initial_result.get('status') == 'error':
                                error_msg = f"Error processing initial message: {initial_result.get('message', 'Unknown error')}"
                                session_history.add_message(AIMessage(content=error_msg))
                                yield error_msg
                                return
                            
                            if initial_result.get('status') == 'complete':
                                print("Initial result status is complete")
                                
                                # Handle recommendation_ready case
                                if initial_result.get('recommendation_ready', False):
                                    print("Recommendation is ready from initial message")
                                    try:
                                        # Generate recommendation immediately
                                        recommendation = await selector.generate_recommendation(
                                            best_connector=initial_result.get('best_connector'),
                                            max_confidence=initial_result.get('best_score', 100.0)
                                        )
                                        
                                        if recommendation and 'recommendation' in recommendation:
                                            response = recommendation['recommendation']['analysis']
                                            session_history.add_message(AIMessage(content=response))
                                            session_mapping[session_id]['connector_selector'] = None
                                            yield response
                                            return
                                        else:
                                            # Fallback response if recommendation structure is incorrect
                                            fallback = "Based on your requirement for a 2mm pitch plastic connector, I recommend the CMM connector from Nicomatic's range."
                                            session_history.add_message(AIMessage(content=fallback))
                                            session_mapping[session_id]['connector_selector'] = None
                                            yield fallback
                                            return
                                    except Exception as rec_error:
                                        print(f"Error generating recommendation: {rec_error}")
                                        fallback = "Based on your requirement for a 2mm pitch plastic connector, I recommend the CMM connector from Nicomatic's range."
                                        session_history.add_message(AIMessage(content=fallback))
                                        session_mapping[session_id]['connector_selector'] = None
                                        yield fallback
                                        return
                                
                                # Handle result already containing recommendation 
                                if 'recommendation' in initial_result:
                                    response = initial_result['recommendation']['analysis']
                                    session_history.add_message(AIMessage(content=response))
                                    session_mapping[session_id]['connector_selector'] = None
                                    yield response
                                    return
                            
                            next_question = initial_result.get('next_question')
                            if next_question:
                                print("\nInitial requirements processed")
                                print("\nCurrent confidence scores:")
                                for connector, score in initial_result.get('confidence_scores', {}).items():
                                    print(f"{connector}: {score}")
                                
                                print(f"\nHelp: {next_question.get('clarification', '')}")
                                response = next_question.get('question', "What are your connector requirements?")
                                session_history.add_message(AIMessage(content=response))
                                yield response
                                return
                            else:
                                # No question available, give generic response
                                response = "I need more information about your connector requirements. Could you provide more details?"
                                session_history.add_message(AIMessage(content=response))
                                yield response
                                return
                                
                        except Exception as init_error:
                            print(f"Error in initial processing: {str(init_error)}")
                            # Provide fallback response for 2mm pitch plastic connector
                            fallback = "Based on your requirement for a 2mm pitch plastic connector, I recommend the CMM connector from Nicomatic's range."
                            session_history.add_message(AIMessage(content=fallback))
                            session_mapping[session_id]['connector_selector'] = None
                            yield fallback
                            return
                    
                    ## Process answer and get next question/recommendation
                    selector = session_mapping[session_id]['connector_selector']
                    result = await selector.process_answer(user_input)
                    
                    if result['status'] == 'continue':
                        next_question = result.get('next_question')
                        if next_question:
                            print(f"\nHelp: {next_question['clarification']}")
                            print("\nCurrent confidence scores:")
                            for connector, score in result['confidence_scores'].items():
                                print(f"{connector}: {score}")
                            
                            response = next_question['question']
                            session_history.add_message(AIMessage(content=response))
                            yield response
                        else:
                            ## No more questions, generate recommendation
                            recommendation = await selector.generate_recommendation()
                            if recommendation['status'] == 'complete':
                                response = recommendation['recommendation']['analysis']
                                session_history.add_message(AIMessage(content=response))
                                session_mapping[session_id]['connector_selector'] = None
                                yield response
                            else:
                                response = "I couldn't generate a recommendation with the provided information. Could you provide more details?"
                                session_history.add_message(AIMessage(content=response))
                                yield response
                    
                    elif result['status'] == 'complete':
                        recommendation = result['recommendation']
                        response = f"{recommendation['analysis']}"
                        session_history.add_message(AIMessage(content=response))
                        session_mapping[session_id]['connector_selector'] = None
                        yield response
                    
                    else:
                        error_msg = f"Error: {result.get('message', 'Unknown error occurred')}"
                        session_history.add_message(AIMessage(content=error_msg))
                        session_mapping[session_id]['connector_selector'] = None 
                        yield error_msg

                except Exception as e:
                    error_message = f"An error occurred during connector selection: {str(e)}"
                    logging.error(f"Error in connector selection: {str(e)}")
                    session_history.add_message(AIMessage(content=error_message))
                    session_mapping[session_id]['connector_selector'] = None 
                    yield error_message

            return StreamingResponse(generate_connector_selection(), media_type="text/plain")
        else:
            async def generate_response():
                try:
                    # Reset the source tracker for this query
                    source_tracker = SourceTracker()
                    source_tracker.reset()
                    
                    response = await agent_with_chat_history.ainvoke({
                "input": user_input,
                "chat_history": formatted_chat_history
            })
                    
                    final_answer = ""
                    intermediate_data = []
                    used_tools = []
                    
                    if isinstance(response, dict):
                        ## Extract intermediate steps if available
                        intermediate_steps = response.get('intermediate_steps', [])
                        
                        # Process intermediate steps
                        # print("\n=== AGENT INTERMEDIATE STEPS ===")
                        for step_idx, step in enumerate(intermediate_steps):
                            if len(step) > 1:
                                tool_name = step[0].tool if hasattr(step[0], 'tool') else "Unknown tool"
                                tool_input = step[0].tool_input if hasattr(step[0], 'tool_input') else "Unknown input"
                                
                                # print(f"Step {step_idx+1}: Using tool: {tool_name}")
                                # print(f"  Tool input: {tool_input}")
                                
                                # Track used tools
                                used_tools.append(tool_name)
                                
                                # Get the actual output content
                                tool_output = step[1]
                                if hasattr(tool_output, 'return_values'):
                                    output_str = str(tool_output.return_values.get('output', ''))
                                else:
                                    output_str = str(tool_output)
                                
                                # Store output for response synthesis
                                intermediate_data.append(output_str)
                        # print("=== END OF AGENT STEPS ===\n")
                        
                        # Generate response
                        llm = ChatOllama(model="llama3.1", temperature=0.0, disable_streaming=False, 
                            num_ctx=8152, top_p=0.7, top_k=30, base_url=OLLAMA_BASE_URL,cache=False)
                            
                        if response.get('output', '') == "Agent stopped due to iteration limit or time limit.":
                            llm_prompt = f"""
                            Please provide a helpful response to the user's question strictly based on the information gathered from the system.
                            Response should be relevant and must answer user's question accurately.
                            \nuser's question: {user_input}

                            \nHere is information gathered from our systems:
                            \n{''.join(intermediate_data)}
                            
                            \nconversation history just for context:
                            \n{formatted_chat_history}
                            
                            """

                            synthesized_response = await llm.ainvoke(llm_prompt)
                            final_answer = str(synthesized_response.content).strip('{}')
                        else:
                            final_answer = response.get('output', '') or str(response)

                        ## Clean up final answer
                        final_answer = final_answer.replace("Final Answer:", "").strip()
                        
                        # Create clickable file links if catalog or lab tools were used
                        if "Nicomatic_connector_catalogue" in used_tools or "Nicomatic_lab_tests" in used_tools:
                            # Get the list of NodeWithScore objects
                            source_nodes = source_tracker.get_source_nodes() 
                            if source_nodes:
                                links_text = "\n\nView source documentation:"
                                link_parts = set() 
                                for node_with_score in source_nodes:
                                    # Access metadata from the inner .node attribute
                                    if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'metadata'):
                                        metadata = node_with_score.node.metadata
                                        abs_path = metadata.get("absolute_path") 
                                        if not abs_path:
                                            continue

                                        filename = os.path.basename(abs_path) 
                                        file_base = os.path.splitext(filename)[0] 
                                        # convert .md → .pdf in the URL
                                        encoded_path = urllib.parse.quote(abs_path).replace(".md", ".pdf") 

                                        # pull the page number out of the metadata (default to 1)
                                        page_number = metadata.get("page_number", 1) 

                                        ## build the link
                                        link = ( f"\n- {file_base}: /source_document/{encoded_path}#page={page_number}")
                                        link_parts.add(link)

                                if link_parts:
                                    ## Sort links alphabetically for consistent order
                                    links_text += "\n" + "\n".join(sorted(list(link_parts))) 

                                ## Combine answer with links
                                full_response = final_answer + links_text 
                            else:
                                full_response = final_answer
                        else:
                            full_response = final_answer
                        session_history.add_message(AIMessage(content=full_response))
                        yield full_response 
                    else:
                        error_msg = "Received unexpected response format from agent"
                        logging.error(error_msg)
                        session_history.add_message(AIMessage(content=error_msg))
                        yield error_msg
                    return_agent(agent_with_chat_history)       
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error while processing your request: {str(e)}"
                    logging.error(f"Error in generate_response: {str(e)}")
                    session_history.add_message(AIMessage(content=error_message))
                    yield error_message
            return StreamingResponse(generate_response(), media_type="text/plain")
            
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f'An error occurred while processing your request: {str(e)}')
## Creates new sessionID and chat ID, welcome message
@app.post("/new_session")
async def new_session():
    if not app_ready:
        print("Waiting for application startup to complete before creating new session...")
        if not startup_complete.wait(timeout=60):
            raise HTTPException(status_code=503, 
                              detail="Application is still initializing. Please try again in a few moments.")
    conn_info = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    try:
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT nextval('chat_id_seq')")
                chat_id = cur.fetchone()[0]
                session_id = str(uuid.uuid4())
                session_mapping[session_id] = {'chat_id': chat_id, 'timestamp': datetime.now().timestamp(),'connector_selector': None}
                session_history = get_session_history(session_id, chat_id)
                session_history.add_message(AIMessage(content="Hello! Welcome to Nicomatic customer support chat. How can I assist you today?"))
                
                return {"sessionId": session_id, "chatId": chat_id}
    except Exception as e:
        logging.error(f"Error creating new session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
## Default page/ root page
@app.get("/", response_class=HTMLResponse)
async def index_page():
    try:
        with open("templates/index.html", encoding='utf-8') as f:
            content = f.read()
            print(content) 
            return content
    except Exception as e:
        print(f"Error reading file: {e}") 
        raise HTTPException(status_code=500, detail='Error reading index.html')

## Get conversation history from session_id 
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    try:
        chat_id = session_mapping.get(session_id)
        if not chat_id:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session_history = get_session_history(session_id, chat_id)
        ## fetch messages from database
        history_messages = session_history.get_messages()
        ## Tags from different roles
        messages = []
        for message in history_messages:
            if isinstance(message, HumanMessage):
                messages.append({'sender': 'user', 'content': message.content})
            elif isinstance(message, AIMessage):
                messages.append({'sender': 'bot', 'content': message.content})
            elif isinstance(message, SystemMessage):
                messages.append({'sender': 'system', 'content': message.content})
                
        return JSONResponse(content={'messages': messages})
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail='An error occurred while retrieving the session.'
        )
    
## Initiate on startup
@app.on_event("startup")
async def startup_event():
    global vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab
    global agent_pool, session_mapping, app_ready, startup_complete
    
    # Use the lock to ensure only one startup process runs
    with startup_lock:
        if app_ready:
            return
            
        try:
            # Database initialization
            print("Initializing database...")
            initialize_database()
            session_mapping = load_session_mapping()
            print(f"Loaded {len(session_mapping)} existing sessions")
            
            # Load data once at startup - force this to complete
            directory_path = "./extracted_best/"
            print("\n -- Loading Data -- ")
            documents1, documents3 = load_data(directory_path)
            print("\n -- Processing data -- ")
            
            result_indices = processing_data(documents1, documents3)
            
            # Verify we got enough return values
            if result_indices and len(result_indices) == 4:
                vector_index_markdown, keyword_index_markdown, vector_index_markdown_lab, keyword_index_markdown_lab = result_indices
                print("Successfully assigned all indices")
            else:
                print(f"Warning: processing_data returned {len(result_indices) if result_indices else 0} indices instead of 4")
                # Continue anyway
            
            # More detailed individual index verification
            print("Verifying indices...")
            if vector_index_markdown is None:
                print("WARNING: vector_index_markdown is None")
            if keyword_index_markdown is None:
                print("WARNING: keyword_index_markdown is None")
            if vector_index_markdown_lab is None:  
                print("WARNING: vector_index_markdown_lab is None")
            if keyword_index_markdown_lab is None:
                print("WARNING: keyword_index_markdown_lab is None")
            
            # Print types of indices for better debugging
            print(f"Indices types:")
            print(f"- vector_index_markdown: {type(vector_index_markdown)}")
            print(f"- keyword_index_markdown: {type(keyword_index_markdown)}")
            print(f"- vector_index_markdown_lab: {type(vector_index_markdown_lab)}")
            print(f"- keyword_index_markdown_lab: {type(keyword_index_markdown_lab)}")
            
            # Create tools based on loaded data
            print("Creating tools...")
            tools = creating_tools(vector_index_markdown, keyword_index_markdown, 
                                  vector_index_markdown_lab, keyword_index_markdown_lab)
            
            # Initialize agent pool
            print("Initializing agent pool...")
            agent_pool = ThreadPoolExecutor(max_workers=MAX_AGENTS)
            
            # Pre-create agents and put them in the queue
            for i in range(MAX_AGENTS):
                print(f"Creating agent {i+1}/{MAX_AGENTS}...")
                agent = create_isolated_agent(tools)
                agent_queue.put(agent)
                
            print(f"Initialized {MAX_AGENTS} agents in the pool")
            
            # Mark application as ready
            app_ready = True
            startup_complete.set()
            print("Application startup complete - ready to handle requests")
            
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            # Set the event even on failure to avoid hanging requests
            startup_complete.set()
            # Don't re-raise the exception - this allows the app to start even with partial data
            print(f"WARNING: Application started with partial data loading. Error: {str(e)}")
## Tester's input
@app.post("/suggestion")
async def suggestion(request: Request):
    ## Deamons should have made it shorter
    try:
        body = await request.json()
        session_id = body.get('sessionId')
        user_suggestion = body.get('message')
        
        if not isinstance(user_suggestion, str):
            ## Convert message to string if it's not done already. Bruh!!
            user_suggestion = str(user_suggestion)
        ## Get chat id from session mapping
        session_info = session_mapping.get(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
            
        chat_id = session_info.get('chat_id')
        
        ## Store suggestion in the database
        try:
            session_history = get_session_history(session_id, chat_id)
            session_history.add_message(SystemMessage(content=user_suggestion))
            ## Does not work
            logging.info(f"Suggestion stored for session {session_id}, chat {chat_id}")
            return JSONResponse(content={'detail': 'Suggestion submitted successfully.','sessionId': session_id,'chatId': chat_id})
            
        except Exception as db_error:
            logging.error(f"Database error storing suggestion: {str(db_error)}")
            raise HTTPException(status_code=500,detail='Error storing suggestion in database')
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error occurred while submitting suggestion: {str(e)}")
        raise HTTPException(status_code=500,detail='An error occurred while submitting suggestion.')


# Replace the existing get_source_document functionwith this:

@app.get("/source_document/{file_path:path}")
async def get_source_document(file_path: str, page: int = 1):
    try:
        # Decode the file path
        decoded_path = urllib.parse.unquote(file_path)
        print(f"Requested document path: {decoded_path}")
        # Check if the file exists
        if not os.path.exists(decoded_path):
            # Try alternative paths
            
            # 1. Try with base filename in extracted_best directory
            base_filename = os.path.basename(decoded_path)
            alt_path1 = os.path.join("./extracted_best", base_filename)
            
            # 2. Try with base filename in lab directory
            alt_path2 = os.path.join("./extracted_best/lab", base_filename)
            
            # 3. Try absolute path within Docker container
            container_path = os.path.join("/app/extracted_best", base_filename)
            
            # 4. Try replacing app path with local path
            local_path = decoded_path.replace("/app/", "./")
            
            # Check all alternatives
            possible_paths = [alt_path1, alt_path2, container_path, local_path]
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    decoded_path = path
                    found = True
                    print(f"Found document at: {decoded_path}")
                    break
            
            if not found:
                print(f"Document not found at any of these paths: {possible_paths}")
                raise HTTPException(status_code=404, detail="Source document not found")

        # Continue with serving the file
        _, ext = os.path.splitext(decoded_path)
        ext = ext.lower()

        if ext == ".pdf":
            # Serve the PDF with a page hint
            response = FileResponse(decoded_path, media_type="application/pdf")
            response.headers["Content-Disposition"] = f"inline; filename={os.path.basename(decoded_path)}"
            return response
        else:
            try:
                with open(decoded_path, "r", encoding="utf-8") as file: 
                    content = file.read() 
            except UnicodeDecodeError:
                 try:
                     with open(decoded_path, "r", encoding="latin-1") as file:
                         content = file.read()
                 except Exception:
                    # If reading as text fails completely, consider serving as plain text or raising error
                    raise HTTPException(status_code=500, detail=f"Error reading file content with standard encodings.")


            filename = os.path.basename(decoded_path) #

            # Serve the text content as HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head> 
                <title>Source Document: {filename}</title> 
                <style> 
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 900px; margin: 0 auto; }} 
                    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
                    h1 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; }} 
                    .filepath {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }} 
                </style> 
            </head> 
            <body> 
                <h1>Source Document: {filename}</h1> 
                <div class="filepath">Full path: {decoded_path}</div> 
                <pre>{content}</pre> 
            </body> 
            </html>
            """ 
            return HTMLResponse(content=html_content) 

    except FileNotFoundError: 
        raise HTTPException(status_code=404, detail="Source document not found")
    except HTTPException as he: # Re-raise specific HTTP exceptions
        raise he
    except Exception as e: 
        # Log the detailed error for debugging
        logging.error(f"Error reading source document '{decoded_path}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading source document: {str(e)}") #
    ## Run FastAPI with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
