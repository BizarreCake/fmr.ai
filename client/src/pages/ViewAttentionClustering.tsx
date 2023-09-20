import {Container, Stack, Typography} from "@mui/material";
import {useParams} from "react-router";
import {useQuery} from "react-query";
import axios from "axios";
import {AttentionHeadPlotEntryWithData} from "../api/types.ts";
import {useLayoutEffect, useRef} from "react";
import * as d3 from 'd3';


interface GetAttentionHeadPlotParams {
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
    enabled: Boolean(params.key),
  });
}


interface HeadPlotProps {
  data: AttentionHeadPlotEntryWithData;
}

function HeadPlot(props: HeadPlotProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  const fullWidth = 600;
  const fullHeight = 600;
  const margin = { top: 15, right: 15, bottom: 15, left: 15 };
  const width = fullWidth - margin.left - margin.right;
  const height = fullHeight - margin.top - margin.bottom;

  useLayoutEffect(() => {
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
      .data(props.data.mds)
      .enter()
      .append('circle')
        .attr('cx', function (d) { return x(d.x); } )
        .attr('cy', function (d) { return y(d.y); } )
        .attr('r', 3)
        .style('fill', '#69b3a2');
  }, [props.data]);

  return (
    <svg
      ref={svgRef}
      width={fullWidth}
      height={fullHeight}
      viewBox={`0 0 ${fullWidth} ${fullHeight}`}
    />
  );
}


export default function ViewAttentionClusteringPage() {
  const {key} = useParams();

  const { data, isLoading } = useGetAttentionHeadPlotQuery({key});

  return (
    <Container maxWidth="lg" sx={{pt: 5}}>
      {data && (
        <Stack
          alignItems="center"
        >
          <HeadPlot data={data.result} />
        </Stack>
      )}
    </Container>
  )
}
