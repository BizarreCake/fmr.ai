import React, {useEffect, useState} from "react";
import axios from "axios";
import {Box, CircularProgress, Stack, Typography, Card, Button, Alert} from "@mui/material";
import {useMutation, useQuery, useQueryClient} from "react-query";

import * as d3 from "d3";
import "d3-graphviz";
import {useNavigate, useParams} from "react-router";
import {useAtomValue} from "jotai";
import {currentModelAtom, useAgentByModelName} from "../state/models.ts";
import {LoadingButton} from "@mui/lab";



interface ModelGraphViewProps {
  dot?: string;
  loading: boolean;
}

function ModelGraphView(props: ModelGraphViewProps) {
  const ref = React.useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const [rendering, setRendering] = useState(false);
  useEffect(() => {
    if (props.dot) {
      const gv = d3.select(ref.current).graphviz();
      setRendering(true);
      gv
        .width('100%')
        .height('100%')
        .zoomScaleExtent([0.1, 50])
        .renderDot(props.dot)
        .fit(true)
        .on('end', () => {
          const svg = d3.select(ref.current).select('svg');

          // clear white background
          svg.select('#graph0').select('polygon').attr('fill', 'transparent');

          const nodeGroups = svg.selectAll('.node');

          // go over comments and extract ids
          console.log('data', gv.data());
          const comments = nodeGroups.selectAll('comment');
          console.log('c', comments);

          // make polygons display yellow border when hovered
          nodeGroups.style('cursor', 'pointer');
          nodeGroups.on('mouseover', function () {
            const polygon = d3.select(this).select('polygon');
            const prevFill = polygon.attr('fill');

            polygon.attr('stroke', 'gold');
            polygon.attr('fill', 'wheat');

            d3.select(this).on('mouseout', function () {
              polygon.attr('stroke', 'black');
              polygon.attr('fill', prevFill);
            });
          });

          // add click handler
          nodeGroups.on('click', function () {
            const nodeData = d3.select(this).data()[0] as { key: string };
            const key = nodeData.key;
            const match = key.match(/id\((.*?)\)/);
            if (match && match.length > 1) {
              const id = match[1];
              const encodedId = encodeURIComponent(id);
              navigate(`/tensor/${encodedId}`);
            }
          });

          setRendering(false);
        });
    }
  }, [props]);

  const loading = props.loading || rendering;

  return (
    <>
      {loading && (
        <Stack
          justifyContent="center"
          alignItems="center"
          spacing={2}
          sx={{
            height: '100%',
            backgroundColor: '#ffffff',
            opacity: 0.8,
            backgroundImage: 'linear-gradient(#f4f4f4 1px, transparent 1px), linear-gradient(to right, #f4f4f4 1px, #ffffff 1px)',
            backgroundSize: '20px 20px',
          }}
        >
          <CircularProgress size={64}/>
          <Typography variant="body2">
            {rendering ? 'Rendering...' : 'Fetching...'}
          </Typography>
        </Stack>
      )}

      <Box
        ref={ref}
        sx={{
          height: '100%',
          overflow: 'hidden',
          backgroundColor: '#ffffff',
          opacity: 0.8,
          backgroundImage: 'linear-gradient(#f4f4f4 1px, transparent 1px), linear-gradient(to right, #f4f4f4 1px, #ffffff 1px)',
          backgroundSize: '20px 20px',
          display: loading ? 'none' : 'block',
        }}>
      </Box>
    </>
  )
}


interface GenerateModelGraphParams {
  project_uuid: string;
  agent_uuid: string;
  model_name: string;
}

interface GenerateModelGraphResponse {

}

function useGenerateModelGraphMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: GenerateModelGraphParams) => {
      const result = await axios.post('/api/model-graph/generate', params);
      return result.data as GenerateModelGraphResponse;
    },
    onSuccess() {
      queryClient.invalidateQueries('model-graph');
    }
  })
}


function NoGraphAvailable() {
  const { projectId } = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);

  const generateGraph = useGenerateModelGraphMutation();
  const handleGenerate = async () => {
    if (!projectId || !agent)
      return;

    await generateGraph.mutateAsync({
      project_uuid: projectId,
      agent_uuid: agent.uuid,
      model_name: agent.model_name,
    });
  };

  return (
    <Stack
      sx={{
        height: '100%',
        backgroundColor: '#ffffff',
        opacity: 0.8,
        backgroundImage: 'repeating-radial-gradient( circle at 0 0, transparent 0, #ffffff 10px ), repeating-linear-gradient( #d8eaff55, #d8eaff )',
      }}
      justifyContent="center"
      alignItems="center"
    >
      <Card sx={{ p: 3 }}>
        {!currentModel && (
          <Alert severity="info">
            There is no model available. Please connect an agent to this project.
          </Alert>
        )}

        {currentModel && (
          <Stack spacing={3}>
            <Typography>
              A model graph of <b>{currentModel}</b> has not been generated yet for this project.
            </Typography>

            <LoadingButton
              variant="contained"
              onClick={handleGenerate}
              loading={generateGraph.isLoading}
            >
              Generate Graph
            </LoadingButton>
          </Stack>
        )}
      </Card>
    </Stack>
  )
}


interface GetModelGraphParams {
  project_uuid: string;
  agent_uuid: string;
}

interface GetModelGraphResponse {
  dot: string;
}

function useGetModelGraphQuery(params: null | GetModelGraphParams) {
  return useQuery(['model-graph', params], async () => {
    const result = await axios.get('/api/model-graph', {params});
    return result.data as GetModelGraphResponse;
  }, {
    refetchOnWindowFocus: false,
    enabled: params !== null,
    retry: false,
  });
}


export default function ViewComputationGraphPage() {
  const { projectId } = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);
  const { data, isLoading} = useGetModelGraphQuery(
    (projectId && agent) ? {
      project_uuid: projectId,
      agent_uuid: agent.uuid,
    } : null
  );

  return (
    <Box
      sx={{
        position: 'absolute',
        inset: 0,
      }}
    >
      {(isLoading || data?.dot) ? (
        <ModelGraphView dot={data?.dot} loading={isLoading} />
      ) : (
        <NoGraphAvailable />
      )}
    </Box>
  );
}
