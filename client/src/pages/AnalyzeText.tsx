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
import {useEffect, useMemo, useRef, useState} from "react";
import {LoadingButton} from "@mui/lab";
import * as d3 from 'd3';
import useResizeObserver from "@react-hook/resize-observer";


interface AnalyzeTextPredictParams {
  text: string;
}

interface TokenizationResult {
  token_ids: number[];
  token_names: string[];
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
  key: string;
  tensor_id: string;
}


interface AttentionHeadExtraction {
  matrix: number[][];
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
        { params, }
      );

      return result.data as AnalyzeTextExtractAttentionResponse;
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


interface AttentionHeadViewProps {
  tensorId: string;
  headIndex: number;
  tokenization: TokenizationResult;
  data: AnalyzeTextExtractAttentionResponse;
}

function AttentionHeadView(props: AttentionHeadViewProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [boxSize, setBoxSize] = useState({width: 0, height: 0});

  const fontSize = 13;
  const lineHeight = fontSize * 2;
  const padding = 16;
  const lineWidth = 60;
  const requiredHeight = useMemo(() => {
    return props.tokenization.token_names.length * lineHeight + 2 * padding;
  }, [props.tokenization]);

  useEffect(() => {
    if (!svgRef.current)
      return;

    svgRef.current.innerHTML = '';  // clear contents

    const svg = d3.select(svgRef.current);

    svg.selectAll('.left')
      .data(props.tokenization.token_names)
      .enter()
      .append('text')
      .attr('font-family', 'monospace')
      .attr('font-size', fontSize)
      .attr('x', padding)
      .attr('y', (_, i) => fontSize + i * lineHeight + padding)
      .text((d) => d);

    svg.selectAll('.right')
      .data(props.tokenization.token_names)
      .enter()
      .append('text')
      .attr('font-family', 'monospace')
      .attr('font-size', fontSize)
      .attr('x', boxSize.width - lineWidth - padding)
      .attr('y', (_, i) => fontSize + i * lineHeight + padding)
      .text((d) => d);

    const attentionMatrix = props.data.batch[0].heads[props.headIndex].matrix;
    svg.selectAll('.lines')
    // iterate over all i,j pairs from 0 to attentionMatrix.length
      .data(Array.from({length: attentionMatrix.length * attentionMatrix[0].length}, (_, v) => ({i: Math.floor(v / attentionMatrix.length), j: v % attentionMatrix.length})))
      .enter()
      .append('line')
      .attr('x1', padding + lineWidth)
      .attr('y1', (d) => (d.i) * lineHeight + padding + fontSize / 2)
      .attr('x2', boxSize.width - 2 * padding - lineWidth)
      .attr('y2', (d) => fontSize / 2 + d.j * lineHeight + padding)
      .attr('stroke', 'blue')
      .attr('stroke-opacity', (d) => attentionMatrix[d.i][d.j])
      .attr('stroke-width', (d) => attentionMatrix[d.i][d.j]);
  }, [props.tokenization, props.data, boxSize]);

  const boxRef = useRef<HTMLDivElement | null>(null);
  useResizeObserver(boxRef, (entry) => {
    setBoxSize({
      width: entry.contentRect.width,
      height: entry.contentRect.height,
    });
  });

  return (
    <Box
      ref={boxRef}
      sx={{
        border: '1px solid #ccc',
        height: requiredHeight,
      }}
    >
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${boxSize.width} ${boxSize.height}`}
      />
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
  const { data, isLoading } = useAnalyzeTextExtractAttentionQuery({
    key: props.cmapKey,
    tensor_id: props.tensorId,
  });

  return (
    <Box>
      <Typography fontWeight="bold" sx={{ mb: 2 }}>Tensor {props.tensorId}</Typography>

      {isLoading && (
        <LinearProgress sx={{ mt: 1 }} />
      )}

      {!isLoading && data && (
        <Grid container spacing={3}>
          {
            Array.from({length: props.instance.num_heads}).map((_, i) => (
              <Grid item xs={4} key={i}>
                <AttentionHeadView
                  tensorId={props.tensorId}
                  headIndex={i}
                  data={data}
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

interface AnalyzeModelFindAttentionResponse {
  instances: AttentionExtractionInstance[];
}

function useAnalyzeModelFindAttentionQuery() {
  return useQuery('model-attention', async () => {
    const result = await axios.get(
      '/api/analyze/model/find_attention',
    );

    return result.data as AnalyzeModelFindAttentionResponse;
  });
}

interface AttentionSectionProps {
  cmapKey: string;
  tokenization: TokenizationResult;
}


function AttentionSection(props: AttentionSectionProps) {
  const {data, isLoading} = useAnalyzeModelFindAttentionQuery();

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2 }}>Attention</Typography>

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

  const [text, setText] = useState('');
  const handleSubmit = async () => {
    const result = await analyzeText.mutateAsync({
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
          {result && <AttentionSection cmapKey={result.key} tokenization={result} />}
        </Stack>
      </Container>
    </Box>
  )
}