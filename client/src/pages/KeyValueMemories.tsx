import {Stack, Box, Button, Card, CardContent, CardHeader, Container, Grid, Typography} from "@mui/material";


function AnalysesSection() {
  return (
    <>
      <Box
        sx={{
          // position: 'absolute',
          // inset: 0,
          flex: 1,
          pt: 3,

          backgroundColor: '#ffffff',
          opacity: 0.8,
          backgroundImage: 'repeating-radial-gradient( circle at 0 0, transparent 0, #ffffff 10px ), repeating-linear-gradient( #d8eaff55, #d8eaff )',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={3}>
            <Grid item xs={4}>
              <Card sx={{p: 3}}>
                <CardHeader title="Find Keys by Example"/>
                <CardContent>
                  <Typography sx={{ mb: 3 }}>
                    Find the keys that give the highest memory coefficients for a set of inputs.
                  </Typography>

                  <Button variant="contained">
                    Try
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Container>
      </Box>

    </>
  )
}


export default function KeyValueMemoriesPage() {
  return (
    <Stack sx={{ height: '100%' }}>
      <Container maxWidth="lg" sx={{pt: 7}}>
        <Typography variant="h4" sx={{mb: 3}}>
          Transformer Key Value Memories
        </Typography>

        <Typography sx={{mb: 5}}>
          The feed-forward layers in a transformer have been shown to store memories in a key-value fashion.
          <br/>
          Transformer feed-forward layers are composed of two linear layers with a some non-linearity (usually ReLU) in
          between.
          <br/>
          The first layer in the feed-forward network represents the keys that perform pattern matching over the
          distribution of values output by the previous self-attention layer.
          <br/>
          The second layer encodes the values, where the final result is a weighted sum of the values based on the memory
          coefficients computed by multiplying the keys with the inputs.
        </Typography>

        <Typography variant="h5" sx={{mb: 3}}>
          Analyses
        </Typography>
      </Container>

      <AnalysesSection/>
    </Stack>
  );
}
