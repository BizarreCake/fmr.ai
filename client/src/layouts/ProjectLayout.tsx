import {AppBar, Box, Container, Toolbar, Typography} from "@mui/material";
import {AppSidebar, SIDEBAR_WIDTH} from "../components/AppSidebar.tsx";
import {PROJECT_SIDEBAR} from "../sidebar.ts";
import {useCurrentProjectQuery} from "../hooks/current.ts";
import {useMemo} from "react";
import {Outlet} from "react-router";


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
  const { data } = useCurrentProjectQuery();

  const sidebarConfig = useMemo(() => ({
    ...PROJECT_SIDEBAR,
    rootPath: `/project/${data?.project?.uuid}`,
  }), [data?.project?.uuid]);

  return (
    <>
      <ProjectAppBar />

      <AppSidebar
        config={sidebarConfig}
        backLink="/projects"
        backText="Back to Projects"
      />

      <Box
        sx={{
          height: 'calc(100vh - 64px)',
          position: 'relative',
          left: SIDEBAR_WIDTH,
          width: `calc(100vw - ${SIDEBAR_WIDTH}px)`,
        }}
      >
        <Outlet/>
      </Box>
    </>
  );
}