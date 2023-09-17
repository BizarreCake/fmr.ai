import {Box, Button, Card, CardHeader, Container, Grid, Typography} from "@mui/material";
import {Link} from "react-router-dom";


export default function ViewModelPage() {
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
          <Typography variant="h4">Model</Typography>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ position: 'relative' }}>
        <Grid container spacing={3} sx={{ mt: 3 }}>
          <Grid item xs={12} lg={6}>
            <Card sx={{ p: 3 }}>
              <Typography variant="subtitle2">Computation Graph</Typography>

              <Typography variant="body2" sx={{ mt: 2 }}>
                View the computation graph of the model, containing the model's parameters and all intermediate activations.
              </Typography>

              <Button
                variant="contained"
                sx={{ mt: 3 }}
                component={Link}
                to="/model/computation-graph"
              >
                Visit
              </Button>

            </Card>
          </Grid>
        </Grid>
      </Container>
    </>
  )
}