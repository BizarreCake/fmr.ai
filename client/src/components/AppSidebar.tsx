import {SidebarConfig, SidebarGroup, SidebarItem} from "../sidebar.ts";
import {useLocation, useNavigate} from "react-router";
import {Link} from "react-router-dom";
import {Box, Divider, List, ListItemButton, ListItemIcon, ListItemText, Typography} from "@mui/material";
import {ArrowBack, SubdirectoryArrowRight} from "@mui/icons-material";


export const SIDEBAR_WIDTH = 300;


interface ViewSidebarItemProps {
  item: SidebarItem;
  indent: number;
  rootPath?: string;
}

function ViewSidebarItem(props: ViewSidebarItemProps) {
  const location = useLocation();
  const fullPath = props.rootPath ? props.rootPath + props.item.path : props.item.path;
  const selected = location.pathname.startsWith(fullPath);

  const navigate = useNavigate();
  const handleClick = () => {
    if (props.item.path)
      navigate(fullPath);
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
          <ViewSidebarItem key={i} item={child} indent={props.indent + 1} rootPath={props.rootPath} />
        ))
      )}
    </>
  );
}


interface ViewSidebarGroupProps {
  rootPath?: string;
  group: SidebarGroup;
}

function ViewSidebarGroup(props: ViewSidebarGroupProps) {
  return (
    <>
      {props.group.title && (
        <>
          <Typography variant="overline" sx={{ml: 2}}>{props.group.title}</Typography>
          <Divider/>
        </>
      )}

      <List>
        {props.group.items.map((item, i) => (
          <ViewSidebarItem key={i} item={item} indent={0} rootPath={props.rootPath} />
        ))}
      </List>
    </>
  )
}


export interface AppSidebarProps {
  config: SidebarConfig;
  backLink?: string;
  backText?: string;
}


export function AppSidebar(props: AppSidebarProps) {
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
      {props.backLink && (
        <List>
          <ListItemButton component={Link} to={props.backLink}>
            <ListItemIcon>
              <ArrowBack />
            </ListItemIcon>
            <ListItemText>
              {props.backText || 'Back'}
            </ListItemText>
          </ListItemButton>
        </List>
      )}

      {props.config.groups.map((group, i) => (
        <ViewSidebarGroup
          key={i}
          group={group}
          rootPath={props.config.rootPath}
        />
      ))}
    </Box>
  )
}


