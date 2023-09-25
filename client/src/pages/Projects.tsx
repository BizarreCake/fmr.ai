import {Container, Typography, Alert, Button} from "@mui/material";
import {useMutation} from "react-query";
import axios from "axios";
import {useState} from "react";
import {NewProjectDialog} from "../dialogs/NewProject.tsx";


function NewProjectButton() {
  const [newProjectOpen, setNewProjectOpen] = useState(false);

  return (
    <>
      <Button variant="contained" size="small" onClick={() => setNewProjectOpen(true)}>
        Create Project
      </Button>

      <NewProjectDialog open={newProjectOpen} onClose={() => setNewProjectOpen(false)} />
    </>
  )
}


export default function ProjectsPage() {
  return (
    <Container maxWidth="lg" sx={{ pt: 5 }}>
      <Typography variant="h4" sx={{ mb: 3 }}>
        Projects
      </Typography>

      <Alert
        severity="info"
        action={
          <NewProjectButton />
        }
        sx={{ pb: 1 }}
      >
        You currently have no projects. Create a new project to get started.
      </Alert>
    </Container>
  )
}