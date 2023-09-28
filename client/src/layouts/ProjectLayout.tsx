import {AppBar, Box, Container, Toolbar, Typography, Select, MenuItem, CircularProgress} from "@mui/material";
import {AppSidebar, SIDEBAR_WIDTH} from "../components/AppSidebar.tsx";
import {PROJECT_SIDEBAR} from "../sidebar.ts";
import {useCurrentProjectQuery} from "../hooks/current.ts";
import {useEffect, useMemo} from "react";
import {Outlet} from "react-router";
import {useListAgentsQuery} from "../api/routes.ts";
import {Agent} from "../api/types.ts";
import {useAtom} from "jotai";
import {currentModelAtom} from "../state/models.ts";


interface ModelSelectionProps {
  projectUuid: string;
}

function collectModels(agents?: Agent[]) {
  if (!agents)
    return [];

  const models = new Set<string>();
  for (const agent of agents) {
    models.add(agent.model_name);
  }

  return Array.from(models);
}

function ModelSelection(props: ModelSelectionProps) {
  const { data, isLoading } = useListAgentsQuery({
    project_uuid: props.projectUuid,
  });

  const modelNames = useMemo(
    () => collectModels(data?.agents),
    [data?.agents]
  );

  const [value, updateValue] = useAtom(currentModelAtom);
  const handleChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    updateValue(event.target.value as string);
  }

  // pick only model if there is only one automatically
  useEffect(() => {
    if (value === null && modelNames.length === 1) {
      updateValue(modelNames[0]);
    }
  }, [modelNames, value]);

  if (isLoading) {
    return (
      <CircularProgress
        size={20}
        sx={{ color: 'white', ml: 1 }}
      />
    );
  }

  return (
    <Select
      value={modelNames.length === 0 ? '__none__' : value}
      sx={{ color: 'white' }}
      disabled={modelNames.length === 0}
      onChange={handleChange}
    >
      {!isLoading && modelNames.length === 0 && (
        <MenuItem value="__none__">None available</MenuItem>
      )}

      {modelNames.map(name => (
        <MenuItem key={name} value={name}>
          {name}
        </MenuItem>
      ))}
    </Select>
  )
}


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
              sx={{ mr: 2, mb: '-2px' }}
              color="#ffffffc0"
            >
              Project
            </Typography>

            <Typography variant="subtitle1">
              {data?.project?.name}
            </Typography>

            {data?.project?.uuid && (
              <>
                <Typography
                  variant="overline"
                  sx={{ ml: 5, mr: 1, mb: '-2px' }}
                  color="#ffffffc0"
                >
                  Model
                </Typography>

                <ModelSelection
                  projectUuid={data.project.uuid}
                />
              </>
            )}
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