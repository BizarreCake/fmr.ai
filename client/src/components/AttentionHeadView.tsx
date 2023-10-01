import {useEffect, useMemo, useRef, useState} from "react";
import * as d3 from "d3";
import useResizeObserver from "@react-hook/resize-observer";
import {Card} from "@mui/material";
import {AttentionHeadExtraction, TokenizationResult} from "../api/types";

export interface AttentionHeadViewProps {
  tokenization: TokenizationResult;
  data: AttentionHeadExtraction;
}

export function AttentionHeadView(props: AttentionHeadViewProps) {
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

    const attentionMatrix = props.data.matrix;
    const seqLen = props.tokenization.token_ids.length;
    svg.selectAll('.lines')
      // iterate over all i,j pairs from 0 to attentionMatrix.length
      .data(Array.from({length: seqLen * seqLen}, (_, v) => ({i: Math.floor(v / attentionMatrix.length), j: v % attentionMatrix.length})))
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
    <Card
      ref={boxRef}
      sx={{
        // border: '1px solid #ccc',
        height: requiredHeight,
      }}
    >
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={`0 0 ${boxSize.width} ${boxSize.height}`}
      />
    </Card>
  )
}
