// Training API client - proxies through FastAPI to Flask service

export interface TrainingData {
  id: string;
  filename: string;
  size: number;
  created: string;
}

export interface TrainingStatus {
  running: boolean;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  logs: string[];
  model_path: string | null;
}

export interface TrainedModel {
  name: string;
  path: string;
  created: string;
}

// Proxy through FastAPI
const PROXY_BASE = '/api/v1/train';

export async function uploadTrainingData(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${PROXY_BASE}/data/upload`, {
    method: 'POST',
    body: formData,
  });
  return response.json();
}

export async function getTrainingDataList(): Promise<TrainingData[]> {
  const response = await fetch(`${PROXY_BASE}/data`);
  const data = await response.json();
  return data.data || [];
}

export async function cleanTrainingData(
  fileId: string,
  config: { min_length?: number; max_length?: number; balance_samples?: boolean }
): Promise<any> {
  const response = await fetch(`${PROXY_BASE}/data/clean`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId, ...config }),
  });
  return response.json();
}

export async function augmentTrainingData(
  fileId: string,
  method: string
): Promise<any> {
  const response = await fetch(`${PROXY_BASE}/data/augment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId, method }),
  });
  return response.json();
}

export async function startTraining(config: {
  data_file: string;
  base_model: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
}): Promise<any> {
  const response = await fetch(`${PROXY_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return response.json();
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const response = await fetch(`${PROXY_BASE}/status`);
  return response.json();
}

export async function stopTraining(): Promise<any> {
  const response = await fetch(`${PROXY_BASE}/stop`, {
    method: 'POST',
  });
  return response.json();
}

export async function getTrainedModels(): Promise<TrainedModel[]> {
  const response = await fetch(`${PROXY_BASE}/models`);
  const data = await response.json();
  return data.data || [];
}

export async function deployModel(modelPath: string): Promise<any> {
  const response = await fetch(`${PROXY_BASE}/deploy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_path: modelPath }),
  });
  return response.json();
}
