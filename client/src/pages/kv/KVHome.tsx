import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Container,
  LinearProgress,
  Stack, Table, TableBody, TableCell, TableHead, TableRow,
  Typography
} from "@mui/material";
import {useQuery} from "react-query";
import axios from "axios";
import {useEffect, useMemo, useRef, useState} from "react";
import {ExpandLess, ExpandMore} from "@mui/icons-material";
import * as d3 from "d3";
import useResizeObserver from "use-resize-observer";
import {atom, useAtomValue, useSetAtom} from "jotai";


const DEFAULT_SIGMA = 5;
const MAX_DOMAIN = 20;
const BOX_SIZE = 16;



interface SelectedNeuron {
  layer: number;
  neuron: number;
}

const selectedNeuronAtom = atom<null | SelectedNeuron>(null);


interface KVStats {
  num_layers: number;
  num_key_neurons_per_layer: {
    [layer: number]: number;
  };
}


interface GetKVInfoParams {
  workflow: string;
}

interface GetKVInfoResponse {
  stats: KVStats;
}

function useGetKVInfoQuery(params: GetKVInfoParams) {
  return useQuery('kv-info', async () => {
    const result = await axios.get(
      '/api/kv/info',
      {params,},
    );
    return result.data as GetKVInfoResponse;
  }, {
    retry: false,
  });
}


function InfoSection() {
  const {} = useGetKVInfoQuery({workflow: import.meta.env.VITE_WORKFLOW_ID})

  return (
    <Box>
      <Typography variant="h5">
        Info
      </Typography>
    </Box>
  )
}


interface KVHeatmap {
  layer: number;
  sigma: number;
  num_neurons: number;
  heatmap: number[];
}

interface GetLayerKeyHeatmapParams {
  workflow: string;
  layer: number;
  sigma: number;
}

interface GetLayerKeyHeatmapResponse {
  heatmap: KVHeatmap;
}

function useGetLayerKeyHeatmapQuery(params: GetLayerKeyHeatmapParams) {
  return useQuery(['kv-layer-key-heatmap', params], async () => {
    const result = await axios.get(
      '/api/kv/keys/heatmap',
      {params,},
    );
    return result.data as GetLayerKeyHeatmapResponse;
  }, {
    retry: false,
  });
}


interface KeyLayerHeatmapProps {
  layer: number;
  sigma: number;
}

function KeyLayerHeatmap(props: KeyLayerHeatmapProps) {
  const {data, isLoading} = useGetLayerKeyHeatmapQuery({
    workflow: import.meta.env.VITE_WORKFLOW_ID,
    layer: props.layer,
    sigma: props.sigma,
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const containerSize = useResizeObserver<HTMLDivElement>({ref: containerRef});

  const [tooltipNeuron, setTooltipNeuron] = useState<number | null>(null);
  const [tooltipSentences, setTooltipSentences] = useState<number | null>(null);
  const setSelectedNeuron = useSetAtom(selectedNeuronAtom);

  useEffect(() => {
    if (!containerRef.current)
      return;

    const container = d3.select(containerRef.current);

    // delete previous svg
    container.select('svg').remove();

    // create new svg
    const svg = container.append('svg')
      .attr('width', '100%')
      .attr('height', '0px')
      .attr('position', 'relative')
      .attr('z-index', 0);

    // create main group
    const mainGroup = svg.append('g')
      .classed('main-group', true);
  }, [containerSize]);

  useEffect(() => {
    if (!data || !containerRef.current)
      return;

    const indexedHeatmap = data.heatmap.heatmap.map((value, i) => [i, value]);

    const container = d3.select(containerRef.current);
    const svg = container.select('svg');
    const mainGroup = svg.select('.main-group');

    const rect = container.node()!.getBoundingClientRect();

    // compute number of rows and columns
    const width = rect.width ?? 0;
    const numCols = Math.floor(width / BOX_SIZE);
    const numRows = Math.ceil(data.heatmap.num_neurons / numCols);

    // set margin and height
    const xMargin = (width - numCols * BOX_SIZE) / 2;
    mainGroup.attr('transform', `translate(${xMargin}, 0)`);
    svg.attr('height', `${numRows * BOX_SIZE}px`);

    // build color scale
    const colorScale = d3.scaleLinear()
      .domain([0, MAX_DOMAIN]).clamp(true)
      .range(['white', 'rgb(25, 118, 210)']);

    // prepare tooltip handlers
    const tooltip = container.select('.tooltip');
    const tooltipMouseOver = (e, d) => {
      tooltip.style('opacity', 1.0);

      // move hover rect
      hoverRect
        .attr('x', (d[0] % numCols) * BOX_SIZE)
        .attr('y', Math.floor(d[0] / numCols) * BOX_SIZE)
        .attr('opacity', 1.0);
    }
    const tooltipMouseLeave = () => {
      tooltip.style('opacity', 0.0);
      hoverRect.attr('opacity', 0.0);
    }
    const tooltipMouseMove = function (e, d) {
      tooltip.style('left', `${e.pageX + 10}px`);
      tooltip.style('top', `${e.pageY - document.documentElement.scrollTop + 10}px`);
      setTooltipNeuron(d[0] + 1);
      setTooltipSentences(d[1]);
    }
    const handleClick = (_, d) => {
      const neuron = d[0];
      const layer = props.layer;
      setSelectedNeuron({layer, neuron});
    }

    // build heatmap
    mainGroup.selectAll('rect')
      .data(indexedHeatmap)
      .enter()
      .append('rect')
      .attr('x', (_, i) => (i % numCols) * BOX_SIZE)
      .attr('y', (_, i) => Math.floor(i / numCols) * BOX_SIZE)
      .attr('width', BOX_SIZE)
      .attr('height', BOX_SIZE)
      .attr('fill', d => colorScale(d[1]))
      .attr('stroke', '#00000010')
      .on('mouseover', tooltipMouseOver)
      .on('mouseleave', tooltipMouseLeave)
      .on('mousemove', tooltipMouseMove)
      .on('click', handleClick);

    const hoverRect = mainGroup.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', BOX_SIZE)
      .attr('height', BOX_SIZE)
      .attr('fill', 'transparent')
      .attr('stroke', 'gold')
      .attr('stroke-width', 3)
      .attr('pointer-events', 'none')
      .attr('opacity', 0.0);
  }, [data, containerSize]);

  return (
    <>
      {isLoading && (
        <LinearProgress/>
      )}

      <Box
        ref={containerRef}
        sx={{
          display: isLoading ? 'none' : 'block',
          overflow: 'visible',
          cursor: 'pointer',
        }}
      >
        <Stack
          className="tooltip"
          sx={{
            zIndex: 100,
            opacity: 0.0,
            position: 'fixed',
            backgroundColor: 'white',
            border: '1px solid black',
            padding: '5px',
            borderRadius: '5px',
          }}
        >
          <Typography variant="caption">
            Neuron <b>#{tooltipNeuron}</b>
          </Typography>
          <Typography variant="caption">
            Sentences: <b>{tooltipSentences}</b> <span style={{color: '#777'}}>(&gt; {DEFAULT_SIGMA}-sigma)</span>
          </Typography>
        </Stack>

      </Box>
    </>
  );
}


interface OneKeyLayerSectionProps {
  layer: number;
  numKeyNeurons: number;
  openInitially?: boolean;
}

function OneKeyLayerSection(props: OneKeyLayerSectionProps) {
  const [open, setOpen] = useState(props.openInitially);

  return (
    <Accordion
      expanded={open}
      onChange={(_, expanded) => setOpen(expanded)}
    >
      <AccordionSummary
        expandIcon={open ? <ExpandLess/> : <ExpandMore/>}
      >
        <Stack direction="row" alignItems="baseline">
          <Typography fontWeight="bold" sx={{minWidth: 80}}>
            Layer {props.layer + 1}
          </Typography>

          <Typography variant="caption" sx={{ml: 2}} color="text.secondary">
            {props.numKeyNeurons} Neurons
          </Typography>
        </Stack>
      </AccordionSummary>

      <AccordionDetails>
        {open && (
          <KeyLayerHeatmap
            layer={props.layer}
            sigma={DEFAULT_SIGMA}
          />
        )}
      </AccordionDetails>
      {/*<List>*/}
      {/*  <ListItemButton sx={{ bgcolor: '#f0f0f0'}}>*/}
      {/*    <ListItemText>*/}
      {/*      Layer {props.layer + 1} Key Neurons*/}
      {/*    </ListItemText>*/}

      {/*    <ListItemIcon sx={{ minWidth: 28 }}>*/}
      {/*      {open ? <ExpandLess onClick={() => setOpen(false)}/> : <ExpandMore onClick={() => setOpen(true)}/>}*/}
      {/*    </ListItemIcon>*/}
      {/*  </ListItemButton>*/}
      {/*</List>*/}
    </Accordion>
  )
}


function KeyLayersSection() {
  const {data, isLoading} = useGetKVInfoQuery({workflow: import.meta.env.VITE_WORKFLOW_ID})

  const reversedEntries = useMemo(() => {
    if (!data)
      return [];

    const entries = Object.entries(data.stats.num_key_neurons_per_layer);
    entries.sort((a, b) => parseInt(b[0]) - parseInt(a[0]));
    return entries;
  }, [data]);

  if (!data || isLoading)
    return <></>;

  return (
    <Box>
      <Typography variant="h5" sx={{mb: 2}}>
        Key Neurons
      </Typography>

      <Box>
        {reversedEntries.map(([layer, numKeyNeurons], i) => (
          <OneKeyLayerSection
            key={layer}
            layer={parseInt(layer)}
            numKeyNeurons={numKeyNeurons}
            openInitially={i === 0}
          />
        ))}
      </Box>
    </Box>
  )
}



interface KVKeySentence {
  sentence_id: number;
  prefix_len: number;
  words: string[];
  value: number;
  z: number;
}

interface GetKeyNeuronDetailsParams {
  workflow: string;
  layer: number;
  neuron: number;
}

interface GetKeyNeuronDetailsResponse {
  sentences: KVKeySentence[];
}

function useGetKeyNeuronDetailsQuery(params: null | GetKeyNeuronDetailsParams) {
  return useQuery(['kv-key-neuron', params], async () => {
    const result = await axios.get(
      '/api/kv/keys/neuron',
      {params,},
    );
    return result.data as GetKeyNeuronDetailsResponse;
  }, {
    retry: false,
    enabled: params !== null,
  });
}


interface KeySentenceTableProps {
  sentences: KVKeySentence[];
}

function KeySentenceTable(props: KeySentenceTableProps) {
  return (
    <Table sx={{ tableLayout: 'fixed' }}>
      <colgroup>
        <col style={{ width: '65%' }} />
      </colgroup>

      <TableHead>
        <TableRow>
          <TableCell>Tokens</TableCell>
          <TableCell>Value</TableCell>
          <TableCell>Z</TableCell>
        </TableRow>
      </TableHead>

      <TableBody>
        {props.sentences.map(sentence => (
          <TableRow
            key={sentence.sentence_id}
            sx={{
              'opacity': Math.abs(sentence.z) < DEFAULT_SIGMA ? 0.6 : 1.0,
            }}
          >
            <TableCell sx={{ fontFamily: 'monospace', wordWrap: 'break-word' }}>
              {sentence.words.map((word, i) => (
                <span
                  key={i}
                  style={{
                    // marginRight: '2px',
                    backgroundColor: sentence.prefix_len >= i ? '#e0e0e0' : 'transparent',
                  }}
                >
                  {word.replace('</w>', ' ')}
                </span>
              ))}
            </TableCell>
            <TableCell>{sentence.value.toFixed(3)}</TableCell>
            <TableCell>{sentence.z.toFixed(3)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function KeyNeuronDetails() {
  const selectedNeuron = useAtomValue(selectedNeuronAtom);

  const { data, isLoading } = useGetKeyNeuronDetailsQuery(!selectedNeuron ? null : {
    workflow: import.meta.env.VITE_WORKFLOW_ID,
    layer: selectedNeuron.layer,
    neuron: selectedNeuron.neuron,
  });

  if (!selectedNeuron)
    return <></>;

  return (
    <Container sx={{pt: 7, px: 3}}>
      <Typography variant="h5" sx={{ mb: 5 }}>
        Neuron {selectedNeuron.neuron + 1} in Layer {selectedNeuron.layer + 1}
      </Typography>

      <Typography variant="h6" sx={{ mb: 3 }}>
        Triggering Sentences
      </Typography>

      {isLoading && <LinearProgress />}
      {data?.sentences && (
        <KeySentenceTable
          sentences={data.sentences}
        />
      )}
    </Container>
  )
}


export function KVHomePage() {
  const selectedNeuron = useAtomValue(selectedNeuronAtom);

  return (
    <Stack direction="row">
      <Box
        sx={{
          flex: 3,
          height: 'calc(100vh - 64px)',
          borderRight: 'solid 1px #ccc',
          overflow: 'auto',
        }}
      >
        <Container
          sx={{
            pt: 7,
            px: 3,
          }}
        >
          <Stack spacing={7}>
            {/*<InfoSection />*/}

            <KeyLayersSection/>
          </Stack>
        </Container>
      </Box>

      {selectedNeuron !== null && (
        <Box sx={{
          flex: 2,
          height: 'calc(100vh - 64px)',
          overflow: 'auto',
        }}>
          <KeyNeuronDetails/>
        </Box>
      )}
    </Stack>
  );
}