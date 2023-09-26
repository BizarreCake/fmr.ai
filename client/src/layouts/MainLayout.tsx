import {AppBar, Box, Container, Toolbar, Typography} from "@mui/material";
import {Outlet} from "react-router";
import {MAIN_SIDEBAR} from "../sidebar.ts";
import {AppSidebar, SIDEBAR_WIDTH} from "../components/AppSidebar.tsx";


function MainAppBar() {
  return (
    <>
      <AppBar position="fixed">
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            <Typography variant="h6" sx={{mr: 5}}>
              fmr.ai
            </Typography>
          </Toolbar>
        </Container>
      </AppBar>

      <Toolbar/>
    </>
  );
}


export function MainLayout() {
  return (
    <>
      <MainAppBar/>
      <AppSidebar config={MAIN_SIDEBAR}/>

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