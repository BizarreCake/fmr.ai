export interface Dataset {
  name: string;
  description: string;
}


export interface AttentionHeadPlotEntry {
  key: string;
  created_at: number;
  dataset_name: string;
  limit: null | number;
}


export interface AttentionHeadPoint {
  x: number;
  y: number;
  tensor_id: string;
  head_index: number;
}

export interface AttentionHeadPlotEntryWithData extends AttentionHeadPlotEntry {
  mds: AttentionHeadPoint[];
}


export interface TokenizationResult {
  token_ids: number[];
  token_names: string[];
}

export interface AttentionHeadExtraction {
  matrix: number[][];
}

export interface AttentionExtraction {
  heads: AttentionHeadExtraction[];
}


export interface Project {
  uuid: string;
  name: string;
  description: null | string;
}


export interface Agent {
  uuid: string;
  name: string;
  description: null | string;
  model_name: string;
}

