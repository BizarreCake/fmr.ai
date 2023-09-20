import {Box, Button, Card, Container, Typography} from "@mui/material";
import {ReactNode} from "react";
import {Link} from "react-router-dom";



export interface FeatureOverviewCardProps {
  title: string;
  description: string;
  link: string;
}

export function FeatureOverviewCard(props: FeatureOverviewCardProps) {
  return (
    <Card sx={{ p: 3 }}>
      <Typography variant="subtitle2">{props.title}</Typography>

      <Typography variant="body2" sx={{ mt: 2 }}>
        {props.description}
      </Typography>

      <Button
        variant="contained"
        sx={{ mt: 3 }}
        component={Link}
        to={props.link}
      >
        Visit
      </Button>
    </Card>
  )
}


export interface TopLevelOverviewProps {
  title: string;
  children?: ReactNode;
}


export function TopLevelOverview(props: TopLevelOverviewProps) {
  return (
    <>
      <Box
        sx={{
          position: 'absolute',
          inset: 0,

          backgroundColor: '#ffffff',
          opacity: 0.8,
          backgroundImage: 'repeating-radial-gradient( circle at 0 0, transparent 0, #ffffff 10px ), repeating-linear-gradient( #d8eaff55, #d8eaff )',
        }}
      />

      <Box
        sx={{
          position: 'relative',
          bgcolor: 'white',
          py: 5,
          borderBottom: 'solid 1px #ddd',
        }}>
        <Container maxWidth="lg">
          <Typography variant="h4">{props.title}</Typography>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{position: 'relative'}}>
        {props.children}
      </Container>
    </>
  )
}