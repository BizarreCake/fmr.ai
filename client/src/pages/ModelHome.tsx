import {Grid} from "@mui/material";
import {FeatureOverviewCard, TopLevelOverview} from "../components/TopLevelOverview";


export default function ModelHomePage() {
  return (
    <>
      <TopLevelOverview
        title="Model"
      >
        <Grid container spacing={3} sx={{mt: 3}}>
          <Grid item xs={12} lg={6}>
            <FeatureOverviewCard
              title="Computation Graph"
              description="View the computation graph of the model, containing the model's parameters and all intermediate activations."
              link="/model/computation-graph"
            />
          </Grid>
        </Grid>
      </TopLevelOverview>
    </>
  )
}