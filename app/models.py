
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime

class Document(BaseModel):
	id: Optional[UUID] = None
	source: str
	uri: str
	title: Optional[str] = None
	metadata: Optional[dict] = Field(default_factory=dict)
	created_at: Optional[datetime] = None

class Chunk(BaseModel):
	id: Optional[UUID] = None
	document_id: UUID
	chunk_index: int
	content: str
	content_tokens: Optional[int] = None
	embedding: Optional[List[float]] = None  # bge-m3: 1024 dims
	tsv: Optional[str] = None
	metadata: Optional[dict] = Field(default_factory=dict)
	created_at: Optional[datetime] = None

class Query(BaseModel):
	question: str
	top_k: int = 5

class RetrievedChunk(BaseModel):
	id: UUID
	content: str
	score: float
	document_id: UUID
	chunk_index: int
	metadata: Optional[dict] = None

class Answer(BaseModel):
	question: str
	answer: str
	context: List[RetrievedChunk]
	citations: Optional[List[Any]] = None
