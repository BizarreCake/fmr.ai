import {Grid} from "@mui/material";
import {FeatureOverviewCard, TopLevelOverview} from "../components/TopLevelOverview";


export default function AttentionHomePage() {
  return (
    <TopLevelOverview
      title="Attention"
    >
      <Grid container spacing={3} sx={{mt: 3}}>
        <Grid item xs={12} lg={6}>
          <FeatureOverviewCard
            title="Attention Head Clustering"
            description="Plot all attention heads in a 2D scatter plot, with distances between points being proportional to the Jensen-Shannon divergence between the attention heads."
            link="/analyze/global/attention/head-clustering"
          />
        </Grid>
      </Grid>
    </TopLevelOverview>
  )
}