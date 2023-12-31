import {Box, CircularProgress, Container, Grid, IconButton, Stack, Typography} from "@mui/material";
import {useParams} from "react-router";
import {useQuery} from "react-query";
import axios from "axios";
import {
  AttentionExtraction,
  AttentionHeadPlotEntryWithData,
  AttentionHeadPoint,
  TokenizationResult
} from "../api/types.ts";
import {createContext, useContext, useEffect, useMemo, useRef, useState} from "react";
import * as d3 from 'd3';
import {ArrowBack} from "@mui/icons-material";
import {Link} from "react-router-dom";
import {AttentionHeadView} from "../components/AttentionHeadView.tsx";
import {useAtomValue} from "jotai";
import {currentModelAtom, useAgentByModelName} from "../state/models.ts";


interface LocalContextValue {
  setRightPaneVisible: (value: boolean) => void;
  selectedHead: null | AttentionHeadPoint;
  setSelectedHead: (value: null | AttentionHeadPoint) => void;
}

const LocalContext = createContext<null | LocalContextValue>(null);


interface GetAttentionHeadPlotParams {
  project_uuid?: string;
  key?: string;
}

interface GetAttentionHeadPlotResponse {
  result: AttentionHeadPlotEntryWithData;
}

function useGetAttentionHeadPlotQuery(params: GetAttentionHeadPlotParams) {
  return useQuery(['attention-head-plot', params.key], async () => {
    const result = await axios.get(
      '/api/analyze/attention/head_plot/get',
      {params,},
    );
    return result.data as GetAttentionHeadPlotResponse;
  }, {
    enabled: Boolean(params.key) && Boolean(params.project_uuid),
  });
}


interface HeadPlotProps {
  data: AttentionHeadPlotEntryWithData;
}

interface HeadPlotState {
  hoveredPoint: null | { x: number, y: number };
  hoveredCircle: null | d3.Selection<SVGCircleElement, unknown, null, undefined>;
}

interface IndexedHeadPoint extends AttentionHeadPoint {
  index: number;
}

function HeadPlot(props: HeadPlotProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  const fullWidth = 600;
  const fullHeight = 600;
  const margin = {top: 32, right: 32, bottom: 16, left: 16};
  const width = fullWidth - margin.left - margin.right;
  const height = fullHeight - margin.top - margin.bottom;

  const indexedPoints: IndexedHeadPoint[] = useMemo(() => {
    return props.data.mds.map((point, i) => ({
      ...point,
      index: i,
    }));
  }, [props.data]);

  const context = useContext(LocalContext);

  useEffect(() => {
    if (!svgRef.current)
      return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const mainGroup = svg
      .append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const xRange = d3.extent(props.data.mds, d => d.x) as [number, number];
    const x = d3.scaleLinear()
      .domain(xRange)
      .range([0, width]);

    const yRange = d3.extent(props.data.mds, d => d.y) as [number, number];
    const y = d3.scaleLinear()
      .domain(yRange)
      .range([0, height]);

    mainGroup.append('g')
      .selectAll('dot')
      .data(indexedPoints)
      .enter()
      .append('circle')
      .attr('cx', function (d) {
        return x(d.x);
      })
      .attr('cy', function (d) {
        return y(d.y);
      })
      .attr('r', 4)
      .style('fill', '#69b3a2');

    const tree = d3.quadtree<IndexedHeadPoint>()
      .extent([[xRange[0], yRange[0]], [xRange[1], yRange[1]]])
      .x(d => x(d.x))
      .y(d => y(d.y))
      .addAll(indexedPoints);

    const state: HeadPlotState = {
      hoveredPoint: null,
      hoveredCircle: null,
    };

    const handleClickPoint = (point: IndexedHeadPoint) => {
      if (context) {
        context.setSelectedHead(point);
        context.setRightPaneVisible(true);
      }
    };

    const onSvgMouseMove = (e: MouseEvent) => {
      // @ts-ignore
      const mouseX = e.layerX - margin.left;
      // @ts-ignore
      const mouseY = e.layerY - margin.top;

      const point = tree.find(mouseX, mouseY, 15);
      if (point !== state.hoveredPoint) {
        if (state.hoveredCircle) {
          state.hoveredCircle.remove();
          state.hoveredPoint = null;
        }

        if (point) {
          state.hoveredPoint = point;
          state.hoveredCircle = mainGroup.append('circle')
            .attr('cx', x(point.x))
            .attr('cy', y(point.y))
            .attr('r', 15)
            .attr('stroke', 'black')
            .attr('stroke-width', 1)
            .attr('fill', 'transparent')
            .on('click', () => handleClickPoint(point))
        }
      }
    };

    svg.on('mousemove', onSvgMouseMove);
  }, [indexedPoints]);

  return (
    <svg
      ref={svgRef}
      width={fullWidth}
      height={fullHeight}
      viewBox={`0 0 ${fullWidth} ${fullHeight}`}
      style={{position: 'relative'}}
    />
  );
}


interface LeftPaneProps {
  fullWidth: boolean;
}

function LeftPane(props: LeftPaneProps) {
  const {projectId, key} = useParams();
  const {data, isLoading} = useGetAttentionHeadPlotQuery({
    project_uuid: projectId,
    key
  });

  return (
    <>
      <Container maxWidth={props.fullWidth ? false : "lg"} sx={{pt: 5}}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{mb: 3}}>
          <IconButton
            component={Link}
            to={-1 as unknown as string}
          >
            <ArrowBack/>
          </IconButton>

          <Typography variant="h5">
            Attention Head Plot
          </Typography>
        </Stack>
      </Container>

      <Stack
        alignItems="center"
        justifyContent="center"
        sx={{
          backgroundColor: '#eeeeee',
          opacity: 0.8,
          backgroundImage: 'linear-gradient(#e6e6e6 1px, transparent 1px), linear-gradient(to right, #e6e6e6 1px, #eeeeee 1px)',
          backgroundSize: '20px 20px',
          height: '600px',
        }}
      >
        {isLoading && <CircularProgress/>}

        {!isLoading && data && (
          <HeadPlot data={data.result}/>
        )}
      </Stack>
    </>
  )
}


interface AnalyzeAttentionHeadInputsParams {
  project_uuid: string;
  agent_uuid: string;
  key: string;
  tensor_id: string;
  head_index: number;
  limit?: number;
}

interface AttentionHeadInput extends TokenizationResult {
  text: string;
}

interface AnalyzeAttentionHeadInputsResponse {
  inputs: AttentionHeadInput[];
  extraction: AttentionExtraction[];
}

function useAnalyzeAttentionHeadInputsQuery(params: null | AnalyzeAttentionHeadInputsParams) {
  return useQuery(['analyze-attention-head-inputs', params], async () => {
    const result = await axios.get(
      '/api/analyze/attention/head_plot/inputs',
      params ? {params,} : {},
    );

    return result.data as AnalyzeAttentionHeadInputsResponse;
  }, {
    enabled: Boolean(params),
  });
}


// interface RightPaneProps {
//
// }


function RightPane() {
  const {projectId, key} = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);
  const context = useContext(LocalContext);

  console.log('rp', context?.selectedHead);

  const {data, isLoading} = useAnalyzeAttentionHeadInputsQuery(
    projectId && agent && key && context?.selectedHead ? {
      project_uuid: projectId,
      agent_uuid: agent.uuid,
      key,
      tensor_id: context.selectedHead.tensor_id,
      head_index: context.selectedHead.head_index,
      limit: 20,
    } : null);

  return (
    <Box sx={{p: 3}}>
      {isLoading && (
        <Stack alignItems="center" spacing={2} sx={{mt: 5}}>
          <CircularProgress/>
          <Typography variant="body2">Loading data...</Typography>
        </Stack>
      )}

      {!isLoading && data && (
        <Grid container spacing={2}>
          {data.inputs.map((input, i) => (
            <Grid item xs={4} key={i}>
              <Typography variant="body2" color="text.secondary" sx={{mb: 1}}>
                {input.text}
              </Typography>

              <AttentionHeadView
                tokenization={input}
                data={data.extraction[i].heads[0]}
              />
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  )
}


export default function ViewAttentionClusteringPage() {
  const [rightPaneVisible, setRightPaneVisible] = useState(false);

  const [selectedHead, setSelectedHead] = useState<null | AttentionHeadPoint>(null);

  const contextValue = useMemo(() => ({
    setRightPaneVisible,
    selectedHead,
    setSelectedHead,
  }), [selectedHead]);

  return (
    <Box
      sx={{
        height: '100%'
      }}
    >
      <LocalContext.Provider value={contextValue}>
        <Stack
          direction="row"
          sx={{height: '100%'}}
        >
          <Box sx={{flex: 1}}>
            <LeftPane
              fullWidth={!rightPaneVisible}
            />
          </Box>

          {rightPaneVisible && (
            <Box sx={{flex: 1, borderLeft: 'solid 1px #ddd', height: '100%', position: 'relative'}}>
              <Box sx={{position: 'absolute', inset: 0, overflow: 'auto'}}>
                <RightPane/>
              </Box>
            </Box>
          )}
        </Stack>
      </LocalContext.Provider>
    </Box>
  )
}
