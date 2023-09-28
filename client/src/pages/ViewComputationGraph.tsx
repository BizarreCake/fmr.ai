import React, {useEffect, useState} from "react";
import axios from "axios";
import {Box, CircularProgress, Stack, Typography, Card, Button, Alert} from "@mui/material";
import {useQuery} from "react-query";

import * as d3 from "d3";
import "d3-graphviz";
import {useNavigate} from "react-router";
import {useAtomValue} from "jotai";
import {currentModelAtom} from "../state/models.ts";


interface ModelGraphResponse {
  dot: string;
}


function useModelGraphQuery() {
  return useQuery('model-graph', async () => {
    const result = await axios.get('/api/model/graph');
    return result.data as ModelGraphResponse;
  }, {
    refetchOnWindowFocus: false,
  });
}


function ModelGraphView() {
  const ref = React.useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const {data, isLoading} = useModelGraphQuery();
  const [rendering, setRendering] = useState(false);
  useEffect(() => {
    if (data?.dot) {
      const gv = d3.select(ref.current).graphviz();
      setRendering(true);
      gv
        .width('100%')
        .height('100%')
        .zoomScaleExtent([0.1, 50])
        .renderDot(data.dot)
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
  }, [data]);

  const loading = isLoading || rendering;

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


function NoGraphAvailable() {
  const currentModel = useAtomValue(currentModelAtom);

  const handleGenerate = () => {

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

            <Button variant="contained" onClick={handleGenerate}>
              Generate Graph
            </Button>
          </Stack>
        )}
      </Card>
    </Stack>
  )
}


export default function ViewComputationGraphPage() {
  return (
    <Box
      sx={{
        position: 'absolute',
        inset: 0,
      }}
    >
      {/*<ModelGraphView/>*/}
      <NoGraphAvailable />
    </Box>
  );
}
