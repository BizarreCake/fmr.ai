import {AppBar, Box, Container, Toolbar, Typography} from "@mui/material";
import {Outlet} from "react-router";


function WorkflowTitle() {
  let title = '';
  switch (import.meta.env.VITE_WORKFLOW_KIND) {
    case 'kv':
      title = 'Key-Value Memories';
      break;
  }

  return (
    <Typography variant="h5">
      {title}
    </Typography>
  )
}


function TempAppBar() {
  return (
    <>
      <AppBar position="fixed">
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            <Typography variant="h6" sx={{mr: 5}}>
              fmr.ai
            </Typography>

            <WorkflowTitle />
          </Toolbar>
        </Container>
      </AppBar>

      <Toolbar/>
    </>
  );
}



export function TempLayout() {
  return (
    <Box>
      <TempAppBar />

      <Outlet />
    </Box>
  )
}
