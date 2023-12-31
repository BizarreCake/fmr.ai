import {
  Box,
  Card,
  CircularProgress,
  Container,
  Grid,
  LinearProgress,
  Stack,
  SxProps,
  TextField,
  Typography
} from "@mui/material";
import axios from "axios";
import {useMutation, useQuery} from "react-query";
import {useState} from "react";
import {LoadingButton} from "@mui/lab";
import {AttentionHeadExtraction, TokenizationResult} from "../api/types";
import {AttentionHeadView} from "../components/AttentionHeadView.tsx";
import {useParams} from "react-router";
import {useAtomValue} from "jotai";
import {currentModelAtom, useAgentByModelName} from "../state/models.ts";


interface AnalyzeTextPredictParams {
  project_uuid: string;
  agent_uuid: string;
  text: string;
}

interface AnalyzeTextPredictResponse extends TokenizationResult {
  key: string;
}

function useAnalyzeTextPredictMutation() {
  return useMutation('analyze-text-predict', async (data: AnalyzeTextPredictParams) => {
    const result = await axios.post(
      '/api/analyze/text/predict',
      data,
    );
    return result.data as AnalyzeTextPredictResponse;
  });
}


interface AnalyzeTextExtractAttentionParams {
  project_uuid?: string;
  agent_uuid?: string;
  key: string;
  tensor_id: string;
}


interface AttentionExtraction {
  heads: AttentionHeadExtraction[];
}

interface AnalyzeTextExtractAttentionResponse {
  batch: AttentionExtraction[];
}


function useAnalyzeTextExtractAttentionQuery(params: AnalyzeTextExtractAttentionParams) {
  return useQuery(
    ['text-attention', params.key, params.tensor_id],
    async () => {
      const result = await axios.get(
        '/api/analyze/text/extract_attention',
        {params,}
      );

      return result.data as AnalyzeTextExtractAttentionResponse;
    }, {
      enabled: params.project_uuid !== null && params.agent_uuid !== null,
    });
}


interface TokenizationSectionProps {
  tokenization: TokenizationResult;
  sx?: SxProps;
}

function TokenizationSection(props: TokenizationSectionProps) {
  return (
    <Box sx={{...props.sx}}>
      <Typography variant="h5">
        Tokenization
      </Typography>

      <Stack direction="row" spacing={3} sx={{mt: 2}}>
        {props.tokenization.token_names.map((name, i) => (
          <Box key={i}>
            <Typography fontFamily="monospace">{name}</Typography>
            <Typography variant="caption" align="center" component="div" color="#666">
              {props.tokenization.token_ids[i]}
            </Typography>
          </Box>
        ))}
      </Stack>
    </Box>
  )
}


interface AttentionLayerViewProps {
  cmapKey: string;
  tensorId: string;
  instance: AttentionExtractionInstance;
  tokenization: TokenizationResult;
}


function AttentionLayerView(props: AttentionLayerViewProps) {
  const { projectId } = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);
  const {data, isLoading} = useAnalyzeTextExtractAttentionQuery({
    project_uuid: projectId,
    agent_uuid: agent?.uuid,
    key: props.cmapKey,
    tensor_id: props.tensorId,
  });

  return (
    <Box>
      <Typography fontWeight="bold" sx={{mb: 2}}>Tensor {props.tensorId}</Typography>

      {isLoading && (
        <LinearProgress sx={{mt: 1}}/>
      )}

      {!isLoading && data && (
        <Grid container spacing={3}>
          {
            Array.from({length: props.instance.num_heads}).map((_, i) => (
              <Grid item xs={4} key={i}>
                <AttentionHeadView
                  data={data.batch[0].heads[i]}
                  tokenization={props.tokenization}
                />
              </Grid>
            ))
          }
        </Grid>
      )}
    </Box>
  )
}


interface AttentionExtractionInstance {
  softmax_value: string;
  num_heads: number;
}

interface AnalyzeModelFindAttentionParams {
  project_uuid: string;
  agent_uuid: string;
}

interface AnalyzeModelFindAttentionResponse {
  instances: AttentionExtractionInstance[];
}

function useAnalyzeModelFindAttentionQuery(params: null | AnalyzeModelFindAttentionParams) {
  return useQuery('model-attention', async () => {
    const result = await axios.get(
      '/api/analyze/model/find_attention',
      { params, }
    );

    return result.data as AnalyzeModelFindAttentionResponse;
  }, {
    enabled: params !== null,
  });
}

interface AttentionSectionProps {
  cmapKey: string;
  tokenization: TokenizationResult;
}


function AttentionSection(props: AttentionSectionProps) {
  const { projectId } = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);
  const {data, isLoading} = useAnalyzeModelFindAttentionQuery(
    projectId && agent ? {
      project_uuid: projectId,
      agent_uuid: agent.uuid,
    } : null
  );

  return (
    <Box>
      <Typography variant="h5" sx={{mb: 2}}>Attention</Typography>

      {isLoading && <CircularProgress sx={{mt: 3}}/>}
      {!isLoading && data && (
        <Stack spacing={3}>
          {data.instances.map(instance => (
            <AttentionLayerView
              key={instance.softmax_value}
              cmapKey={props.cmapKey}
              tensorId={instance.softmax_value}
              instance={instance}
              tokenization={props.tokenization}
            />
          ))}
        </Stack>
      )}
    </Box>
  )
}


export default function AnalyzeTextPage() {
  const analyzeText = useAnalyzeTextPredictMutation();
  const [result, setResult] = useState<AnalyzeTextPredictResponse | null>(null);

  const { projectId } = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);

  const [text, setText] = useState('');
  const handleSubmit = async () => {
    if (!projectId || !agent)
      return;

    const result = await analyzeText.mutateAsync({
      project_uuid: projectId,
      agent_uuid: agent.uuid,
      text,
    });

    setResult(result);
  };

  return (
    <Box sx={{p: 3}}>
      <Container maxWidth="lg">
        <Card sx={{p: 3}}>
          <TextField
            fullWidth
            placeholder="Enter text to analyze..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <LoadingButton
            variant="contained"
            sx={{mt: 2, ml: 'auto', mr: 0, display: 'table'}}
            onClick={handleSubmit}
            loading={analyzeText.isLoading}
          >
            Submit
          </LoadingButton>
        </Card>

        <Stack spacing={7} sx={{mt: 7}}>
          {result && <TokenizationSection tokenization={result}/>}
          {result && <AttentionSection cmapKey={result.key} tokenization={result}/>}
        </Stack>
      </Container>
    </Box>
  )
}