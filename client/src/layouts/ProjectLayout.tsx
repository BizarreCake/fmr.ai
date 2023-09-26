import {AppBar, Container, Toolbar, Typography} from "@mui/material";
import {AppSidebar} from "../components/AppSidebar.tsx";
import {PROJECT_SIDEBAR} from "../sidebar.ts";
import {useCurrentProjectQuery} from "../hooks/current.ts";


function ProjectAppBar() {
  const { data } = useCurrentProjectQuery();

  return (
    <>
      <AppBar position="fixed">
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            <Typography variant="h6" sx={{mr: 5}}>
              fmr.ai
            </Typography>

            <Typography
              variant="overline"
              sx={{ mr: 1, mb: '-2px' }}
              color="#ffffffc0"
            >
              Project
            </Typography>

            <Typography variant="subtitle1">
              {data?.project?.name}
            </Typography>
          </Toolbar>
        </Container>
      </AppBar>

      <Toolbar/>
    </>
  );
}


export function ProjectLayout() {
  return (
    <>
      <ProjectAppBar />
      <AppSidebar
        config={PROJECT_SIDEBAR}
        backLink="/projects"
        backText="Back to Projects"
      />
    </>
  );
}