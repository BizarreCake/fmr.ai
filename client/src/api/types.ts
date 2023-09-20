

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
}

export interface AttentionHeadPlotEntryWithData extends AttentionHeadPlotEntry {
  mds: AttentionHeadPoint[];
}

