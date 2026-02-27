// Document types
export interface Document {
  id: string;
  text: string;
  file_path: string;
  file_type: string;
  metadata?: {
    chunk_index?: number;
    file_mtime?: number;
    [key: string]: unknown;
  };
  tags?: string[];
}

// Search types
export interface SearchResult {
  id: string;
  text: string;
  file_path: string;
  file_type: string;
  distance?: number;
  score?: number;
  metadata?: {
    chunk_index?: number;
    file_mtime?: number;
    [key: string]: unknown;
  };
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  hybrid?: boolean;
  use_bm25?: boolean;
}

export interface SearchResponse {
  success: boolean;
  results: SearchResult[];
  query: string;
  total: number;
}

// Stats types
export interface Stats {
  total_documents: number;
  total_chunks: number;
  file_types: Record<string, number>;
  storage_size: number;
}

// Upload types
export interface UploadResponse {
  success: boolean;
  message: string;
  task_id?: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  message?: string;
  result?: {
    documents_count: number;
    chunks_count: number;
  };
}

// Delete types
export interface DeleteResponse {
  success: boolean;
  message: string;
  deleted_count?: number;
}

// Tag types
export interface Tag {
  id: string;
  name: string;
  color: string;
  created_at?: string;
  updated_at?: string;
}

export interface CreateTagRequest {
  name: string;
  color: string;
}

export interface UpdateTagRequest {
  name?: string;
  color?: string;
}

export interface DocumentTagsResponse {
  success: boolean;
  tags: Tag[];
}

// Export/Import types
export interface ExportRequest {
  format: 'json' | 'csv' | 'md';
}

export interface ImportRequest {
  format: 'json' | 'csv';
}

// API Response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}
