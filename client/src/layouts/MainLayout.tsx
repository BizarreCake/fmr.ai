import {
  AppBar,
  Box,
  Container,
  Divider,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography
} from "@mui/material";
import {Outlet, useLocation, useNavigate} from "react-router";
import {MAIN_SIDEBAR, SidebarGroup, SidebarItem} from "../sidebar.ts";
import {SubdirectoryArrowRight} from "@mui/icons-material";


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


const SIDEBAR_WIDTH = 300;


interface ViewSidebarItemProps {
  item: SidebarItem;
  indent: number;
}

function ViewSidebarItem(props: ViewSidebarItemProps) {
  const location = useLocation();
  const selected = location.pathname.startsWith(props.item.path);

  const navigate = useNavigate();
  const handleClick = () => {
    if (props.item.path)
      navigate(props.item.path);
  };

  return (
    <>
      <ListItemButton selected={selected} onClick={handleClick}>
        {props.indent > 0 && (
          <ListItemIcon>
            <SubdirectoryArrowRight sx={{width: 20, ml: 2, mt: -0.5}}/>
          </ListItemIcon>
        )}
        <ListItemText primaryTypographyProps={{variant: props.indent === 0 ? 'body1' : 'body2'}}>
          {props.item.text}
        </ListItemText>
      </ListItemButton>

      {props.item.children && (
        props.item.children.map((child, i) => (
          <ViewSidebarItem key={i} item={child} indent={props.indent + 1}/>
        ))
      )}
    </>
  );
}


interface ViewSidebarGroupProps {
  group: SidebarGroup;
}

function ViewSidebarGroup(props: ViewSidebarGroupProps) {
  return (
    <>
      <Typography variant="overline" sx={{ml: 2}}>{props.group.title}</Typography>
      <Divider/>

      <List>
        {props.group.items.map((item, i) => (
          <ViewSidebarItem key={i} item={item} indent={0}/>
        ))}
      </List>
    </>
  )
}


function MainSidebar() {
  return (
    <Box
      sx={{
        bgcolor: '#f8f8f8',
        position: 'fixed',
        width: SIDEBAR_WIDTH,
        height: 'calc(100vh - 64px)',
        borderRight: 'solid 1px #ddd',
        pt: 2,
      }}
    >
      {MAIN_SIDEBAR.groups.map((group, i) => (
        <ViewSidebarGroup key={i} group={group}/>
      ))}
    </Box>
  )
}


export function MainLayout() {
  return (
    <>
      <MainAppBar/>
      <MainSidebar/>

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