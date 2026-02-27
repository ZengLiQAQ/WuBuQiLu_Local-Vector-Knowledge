import type {
  SearchRequest,
  SearchResponse,
  Stats,
  UploadResponse,
  TaskStatus,
  DeleteResponse,
  Tag,
  CreateTagRequest,
  UpdateTagRequest,
  Document,
} from '@/types';

const API_BASE = '/api/v1';

// Helper function to get API key from localStorage
function getApiKey(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('api_key');
  }
  return null;
}

// Helper function to build headers with API key
function getHeaders(): HeadersInit {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };
  const apiKey = getApiKey();
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
}

// ==================== Search API ====================

export async function search(request: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(request),
  });
  return response.json();
}

export async function batchSearch(
  queries: string[],
  topK: number = 5
): Promise<{ success: boolean; results: Record<string, SearchResponse> }> {
  const response = await fetch(`${API_BASE}/batch_search`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ queries, top_k: topK }),
  });
  return response.json();
}

// ==================== Stats API ====================

export async function getStats(): Promise<Stats> {
  const response = await fetch(`${API_BASE}/stats`, {
    headers: getHeaders(),
  });
  return response.json();
}

// ==================== Upload API ====================

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const apiKey = getApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const response = await fetch(`${API_BASE}/upload_async`, {
    method: 'POST',
    headers,
    body: formData,
  });
  return response.json();
}

export async function uploadFiles(files: FileList): Promise<UploadResponse> {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  const apiKey = getApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const response = await fetch(`${API_BASE}/batch_upload`, {
    method: 'POST',
    headers,
    body: formData,
  });
  return response.json();
}

export async function getTaskStatus(taskId: string): Promise<TaskStatus> {
  const response = await fetch(`${API_BASE}/task/${taskId}`, {
    headers: getHeaders(),
  });
  return response.json();
}

// ==================== Delete API ====================

export async function deleteDocument(documentId: string): Promise<DeleteResponse> {
  const response = await fetch(`${API_BASE}/delete`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ document_id: documentId }),
  });
  return response.json();
}

export async function deleteDocuments(documentIds: string[]): Promise<DeleteResponse> {
  const response = await fetch(`${API_BASE}/batch_delete`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ document_ids: documentIds }),
  });
  return response.json();
}

export async function clearKnowledgeBase(): Promise<DeleteResponse> {
  const response = await fetch(`${API_BASE}/clear`, {
    method: 'POST',
    headers: getHeaders(),
  });
  return response.json();
}

// ==================== Tag API ====================

export async function getTags(): Promise<{ success: boolean; tags: Tag[] }> {
  const response = await fetch(`${API_BASE}/tags`, {
    headers: getHeaders(),
  });
  return response.json();
}

export async function createTag(request: CreateTagRequest): Promise<{ success: boolean; tag: Tag }> {
  const response = await fetch(`${API_BASE}/tags`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify(request),
  });
  return response.json();
}

export async function updateTag(tagId: string, request: UpdateTagRequest): Promise<{ success: boolean; tag: Tag }> {
  const response = await fetch(`${API_BASE}/tags/${tagId}`, {
    method: 'PUT',
    headers: getHeaders(),
    body: JSON.stringify(request),
  });
  return response.json();
}

export async function deleteTag(tagId: string): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/tags/${tagId}`, {
    method: 'DELETE',
    headers: getHeaders(),
  });
  return response.json();
}

export async function getDocumentsByTag(tagId: string): Promise<{ success: boolean; documents: Document[] }> {
  const response = await fetch(`${API_BASE}/tags/${tagId}/documents`, {
    headers: getHeaders(),
  });
  return response.json();
}

export async function addTagToDocument(
  documentId: string,
  tagId: string
): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/tags`, {
    method: 'POST',
    headers: getHeaders(),
    body: JSON.stringify({ tag_id: tagId }),
  });
  return response.json();
}

export async function removeTagFromDocument(
  documentId: string,
  tagId: string
): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/tags/${tagId}`, {
    method: 'DELETE',
    headers: getHeaders(),
  });
  return response.json();
}

export async function setDocumentTags(
  documentId: string,
  tagIds: string[]
): Promise<{ success: boolean; message: string }> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/tags`, {
    method: 'PUT',
    headers: getHeaders(),
    body: JSON.stringify({ tag_ids: tagIds }),
  });
  return response.json();
}

// ==================== Export/Import API ====================

export async function exportData(
  format: 'json' | 'csv' | 'md'
): Promise<Blob> {
  const response = await fetch(`${API_BASE}/export?format=${format}`, {
    headers: getHeaders(),
  });
  return response.blob();
}

export async function importData(
  file: File,
  format: 'json' | 'csv'
): Promise<{ success: boolean; message: string; imported_count?: number }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('format', format);

  const apiKey = getApiKey();
  const headers: HeadersInit = {};
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const response = await fetch(`${API_BASE}/import`, {
    method: 'POST',
    headers,
    body: formData,
  });
  return response.json();
}

// ==================== Settings API ====================

export function setApiKey(key: string): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem('api_key', key);
  }
}

export function clearApiKey(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('api_key');
  }
}

export function getStoredApiKey(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('api_key');
  }
  return null;
}
