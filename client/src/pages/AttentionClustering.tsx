import {
  Box,
  Button,
  CircularProgress,
  Container,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Stack,
  Typography
} from "@mui/material";
import {useState} from "react";
import {ChooseDatasetDialog} from "../dialogs/ChooseDataset.tsx";
import {AttentionHeadPlotEntry, Dataset} from "../api/types.ts";
import {useMutation, useQuery, useQueryClient} from "react-query";
import axios from "axios";
import {Add, ScatterPlot} from "@mui/icons-material";
import {formatDistance} from "date-fns";
import {Link} from "react-router-dom";
import {useParams} from "react-router";
import {useAtomValue} from "jotai";
import {currentModelAtom, useAgentByModelName} from "../state/models.ts";


interface ComputeDistanceMatrixParams {
  project_uuid: string;
  agent_uuid: string;
  dataset_name: string;
  limit?: number;
}

interface ComputeDistanceMatrixResponse {

}

function useComputeDistanceMatrixMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: ComputeDistanceMatrixParams) => {
      const result = await axios.post(
        '/api/analyze/attention/compute_attention_head_plot',
        params,
      );

      return result.data as ComputeDistanceMatrixResponse;
    },
    onSuccess: () => {
      queryClient.invalidateQueries('list-attention-head-plots');
    },
  });
}


function AnalysisRunner() {
  const [chooseDatasetDialogOpen, setChooseDatasetDialogOpen] = useState(false);

  const {projectId} = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);
  const computeDistanceMatrixMutation = useComputeDistanceMatrixMutation();

  const handleSelectDataset = (dataset: Dataset, limit?: number) => {
    if (!projectId || !agent)
      return;

    setChooseDatasetDialogOpen(false);

    computeDistanceMatrixMutation.mutate({
      project_uuid: projectId,
      agent_uuid: agent.uuid,
      dataset_name: dataset.name,
      limit: 100,
    });
  };

  const handleRunAnalysis = () => {
    setChooseDatasetDialogOpen(true);
  };

  return (
    <>
      <ChooseDatasetDialog
        open={chooseDatasetDialogOpen}
        onClose={() => setChooseDatasetDialogOpen(false)}
        onSelect={handleSelectDataset}
      />

      <List>
        <ListItemButton
          sx={{border: 'dashed 1px #ccc', borderRadius: '8px'}}
          onClick={handleRunAnalysis}>
          <ListItemIcon><Add/></ListItemIcon>
          <ListItemText>Run Analysis</ListItemText>
        </ListItemButton>
      </List>
    </>
  )
}


interface ListAttentionHeadPlotsParams {
  project_uuid: string;
  agent_uuid: string;
}

interface ListAttentionHeadPlotsResponse {
  items: AttentionHeadPlotEntry[];
}

function useListAttentionHeadPlotsQuery(params: null | ListAttentionHeadPlotsParams) {
  return useQuery('list-attention-head-plots', async () => {
    const result = await axios.get(
      '/api/analyze/attention/head_plot/list',
      { params, },
    );
    return result.data as ListAttentionHeadPlotsResponse;
  }, {
    enabled: params !== null,
  });
}

function AnalysisListSection() {
  const { projectId } = useParams();
  const currentModel = useAtomValue(currentModelAtom);
  const agent = useAgentByModelName(currentModel);
  const {data, isLoading} = useListAttentionHeadPlotsQuery(
    projectId && agent ? {
      project_uuid: projectId,
      agent_uuid: agent.uuid,
    } : null
  );

  return (
    <>
      <Typography variant="h6" sx={{mb: 2}}>
        Analyses
      </Typography>

      {isLoading && (
        <Stack direction="row" spacing={2} alignItems="center">
          <Typography variant="body2">Loading results...</Typography>
          <CircularProgress size={16}/>
        </Stack>
      )}

      {!isLoading && data && (
        <List>
          {data.items.map(item => (
            <ListItemButton
              key={item.key}
              component={Link}
              to={`/project/${projectId}/analysis/global/attention/head-clustering/${item.key}`}
            >
              <ListItemIcon>
                <ScatterPlot/>
              </ListItemIcon>

              <ListItemText>
                <Box>
                  <Typography variant="subtitle2">Dataset: {item.dataset_name}</Typography>
                  <Typography variant="body2">Number of data points: {item.limit}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Created {formatDistance(new Date(item.created_at * 1000), new Date(), {addSuffix: true})}
                  </Typography>
                </Box>
              </ListItemText>
            </ListItemButton>
          ))}
        </List>
      )}

      <Box/>
    </>
  )
}


export default function AttentionClusteringPage() {
  return (
    <>
      <Container maxWidth="lg" sx={{pt: 7}}>
        <Typography variant="h4" sx={{ mb: 3 }}>
          Attention Head Clustering
        </Typography>

        <Typography sx={{mt: 1, mb: 5}}>
          Visualize the relationships between attention heads by plotting them in a 2D scatter plot.
          <br/>
          The distributions of the attention heads are estimated by evaluating the model on a number of inputs from a
          dataset of your choice.
          <br/>
          Jensen-Shannon divergence is used to compute the distances between the distributions, and MDS is used to
          project the attention heads into 2D space.
        </Typography>

        <AnalysisListSection/>

        <AnalysisRunner/>
      </Container>
    </>
  );
}