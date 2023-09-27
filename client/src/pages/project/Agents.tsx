import {CircularProgress, Container, Typography, List, ListItemButton, ListItemText, ListItemIcon} from "@mui/material";
import {Agent} from "../../api/types.ts";
import {useQuery} from "react-query";
import axios from "axios";
import {useCurrentProjectQuery} from "../../hooks/current.ts";
import {Add} from "@mui/icons-material";
import {useState} from "react";
import {NewAgentDialog} from "../../dialogs/NewAgent.tsx";


interface ListAgentsParams {
  project_uuid: string;
}

interface ListAgentsResponse {
  agents: Agent[];
}

function useListAgentsQuery(params?: ListAgentsParams) {
  return useQuery(['list-agents', params], async () => {
    const result = await axios.get('/api/projects/agents/list', {
      params,
    });
    return result.data as ListAgentsResponse;
  }, {
    enabled: !!params?.project_uuid,
  });
}


function AddAgentListButton() {
  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <>
      <NewAgentDialog open={dialogOpen} onClose={() => setDialogOpen(false)} />

      <ListItemButton
        sx={{ border: 'dashed 1px #ccc', borderRadius: '8px' }}
        onClick={() => setDialogOpen(true)} >
        <ListItemIcon><Add /></ListItemIcon>
        <ListItemText>Add Agent</ListItemText>
      </ListItemButton>
    </>
  )
}


function AgentList() {
  const {data: projectData} = useCurrentProjectQuery();
  const {data, isLoading} = useListAgentsQuery(projectData?.project?.uuid ? {
    project_uuid: projectData?.project?.uuid,
  } : undefined);

  return (
    <>
      {isLoading && <CircularProgress sx={{ display: 'table', mx: 'auto' }} />}
      {!isLoading && data && (
        <>
          <List>
            <AddAgentListButton />
          </List>
        </>
      )}
    </>
  );
}


export default function ProjectAgentsPage() {
  return (
    <Container maxWidth="lg" sx={{pt: 7}}>
      <Typography variant="h4" sx={{mb: 3}}>
        Agents
      </Typography>

      <Typography>
        Agents are servers that run a single model (as of this moment) and provide an API for interacting with it.
      </Typography>

      <Typography sx={{mb: 5}}>
        In addition, projects can be associated with multiple agents for conducting multi-model experiments.
      </Typography>

      <AgentList/>
    </Container>
  )
}