import {
  Card,
  Chip,
  CircularProgress,
  Container,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Stack,
  Typography
} from "@mui/material";
import {useCurrentProjectQuery} from "../../hooks/current.ts";
import {Add} from "@mui/icons-material";
import {useState} from "react";
import {NewAgentDialog} from "../../dialogs/NewAgent.tsx";
import {useListAgentsQuery} from "../../api/routes.ts";


interface AddAgentListButtonProps {
  projectUuid?: string;
}

function AddAgentListButton(props: AddAgentListButtonProps) {
  const [dialogOpen, setDialogOpen] = useState(false);

  return (
    <>
      <NewAgentDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        projectUuid={props.projectUuid}
      />

      <ListItemButton
        sx={{border: 'dashed 1px #ccc', borderRadius: '8px'}}
        onClick={() => setDialogOpen(true)}>
        <ListItemIcon><Add/></ListItemIcon>
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
      {isLoading && <CircularProgress sx={{display: 'table', mx: 'auto'}}/>}
      {!isLoading && data && (
        <>
          <Stack spacing={2} sx={{mb: 2}}>
            {data.agents.map(agent => (
              <Card key={agent.uuid} sx={{p: 3}}>
                <table>
                  <colgroup>
                    <col span="1" style={{minWidth: 100}}/>
                  </colgroup>

                  <tbody>
                  <tr>
                    <td><Typography variant="overline">Agent</Typography></td>
                    <td><Typography>{agent.name}</Typography></td>
                  </tr>
                  <tr>
                    <td><Typography variant="overline">Model</Typography></td>
                    <td><Typography>{agent.model_name}</Typography></td>
                  </tr>
                  <tr>
                    <td><Typography variant="overline">Status</Typography></td>
                    <td><Chip color="success" label="Available" size="small"/></td>
                  </tr>
                  </tbody>
                </table>
              </Card>
            ))}
          </Stack>

          <List>
            <AddAgentListButton
              projectUuid={projectData?.project?.uuid}
            />
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