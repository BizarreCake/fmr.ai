import {Alert, Button, Container, List, ListItemButton, ListItemIcon, ListItemText, Typography} from "@mui/material";
import {useState} from "react";
import {NewProjectDialog} from "../dialogs/NewProject.tsx";
import {Project} from "../api/types.ts";
import {useQuery} from "react-query";
import axios from "axios";
import {Add} from "@mui/icons-material";
import {useNavigate} from "react-router";


function NewProjectAlertButton() {
  const [newProjectOpen, setNewProjectOpen] = useState(false);

  return (
    <>
      <Button variant="contained" size="small" onClick={() => setNewProjectOpen(true)}>
        Create Project
      </Button>

      <NewProjectDialog open={newProjectOpen} onClose={() => setNewProjectOpen(false)}/>
    </>
  )
}


function NewProjectListButton() {
  const [newProjectOpen, setNewProjectOpen] = useState(false);

  return (
    <>
      <ListItemButton
        sx={{ border: 'dashed 1px #ccc', borderRadius: '8px', mt: 1 }}
        onClick={() => setNewProjectOpen(true)}
      >
        <ListItemIcon>
          <Add />
        </ListItemIcon>
        <ListItemText>
          Create New Project
        </ListItemText>
      </ListItemButton>

      <NewProjectDialog open={newProjectOpen} onClose={() => setNewProjectOpen(false)}/>
    </>
  )
}


interface ListProjectsParams {}
interface ListProjectsResponse {
  projects: Project[];
}

function useListProjectsQuery(params: ListProjectsParams) {
  return useQuery({
    queryKey: ['projects'],
    queryFn: async () => {
      const response = await axios.get('/api/projects/list', {params});
      return response.data as ListProjectsResponse;
    }
  })
}

function ProjectList() {
  const { data, isLoading } = useListProjectsQuery({});

  const navigate = useNavigate();
  const handleClickProject = (project: Project) => {
    navigate(`/project/${project.uuid}`);
  };

  return (
    <>
      {!isLoading && data && data.projects.length === 0 && (
        <Alert severity="info"
          action={
            <NewProjectAlertButton/>
          }
          sx={{pb: 1}}
        >
          You currently have no projects. Create a new project to get started.
        </Alert>
      )}

      {!isLoading && data && data.projects.length > 0 && (
        <List>
          {data.projects.map(project => (
            <ListItemButton key={project.uuid} onClick={() => handleClickProject(project)}>
              <ListItemText>
                {project.name}
              </ListItemText>
            </ListItemButton>
          ))}

          <NewProjectListButton />
        </List>
      )}

    </>
  );
}


export default function ProjectsPage() {
  return (
    <Container maxWidth="lg" sx={{pt: 5}}>
      <Typography variant="h4" sx={{mb: 3}}>
        Projects
      </Typography>

      <ProjectList />
    </Container>
  )
}