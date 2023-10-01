import {
  Box,
  Card,
  Chip,
  Container,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stack,
  Typography
} from "@mui/material";
import {Article, CheckBox, CheckBoxOutlineBlank} from "@mui/icons-material";


enum PaperFeatureStatus {
  IMPLEMENTED = 'implemented',
  NOT_IMPLEMENTED = 'not_implemented',
  COMING_SOON = 'coming_soon',
}


interface PaperFeature {
  title: string;
  status: PaperFeatureStatus;
}

interface Paper {
  title: string;
  link: string;
  authors: string[];
  features: PaperFeature[];
}

interface PaperCardProps extends Paper {
}


const PAPERS: Paper[] = [
  {
    title: "What does BERT Look At? An Analysis of BERT's Attention",
    link: "https://arxiv.org/abs/1906.04341",
    authors: ['Kevin Clark', 'Urvashi Khandelwal', 'Omer Levy', 'Christopher D. Manning'],
    features: [
      {
        title: "Attention head clustering",
        status: PaperFeatureStatus.IMPLEMENTED,
      },
    ],
  },
];


function PaperCard(props: PaperCardProps) {
  return (
    <Card sx={{p: 3}}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Article/>
        <Typography variant="subtitle2" component="a" href={props.link} target="_blank">
          {props.title}
        </Typography>
      </Stack>

      <Typography variant="body2">
        {props.authors.join(', ')}
      </Typography>

      <Divider sx={{mt: 2}}/>
      <Typography variant="overline">Implemented features:</Typography>

      <List dense>
        {props.features.map((feature, i) => (
          <ListItem key={i}>
            <ListItemIcon>
              {feature.status === PaperFeatureStatus.IMPLEMENTED ? (
                <CheckBox/>
              ) : (
                <CheckBoxOutlineBlank/>
              )}
            </ListItemIcon>
            <ListItemText>
              Attention head clustering

              {feature.status === PaperFeatureStatus.COMING_SOON &&
                <Chip label="Coming soon" size="small" sx={{ml: 2}}/>}
            </ListItemText>
          </ListItem>
        ))}
      </List>
    </Card>
  )
}


function PapersSection() {
  return (
    <Box>
      <Typography variant="h5" sx={{mt: 5, mb: 2}}>
        Papers
      </Typography>

      <Typography>
        We hope to implement and include as many useful results from papers about interpretability and explainability.
      </Typography>

      <Typography>
        Here are some papers whose results we currently implement (at least partially):
      </Typography>

      <Stack spacing={3} sx={{mt: 3}}>
        {PAPERS.map((paper, i) => (
          <PaperCard key={i} {...paper} />
        ))}
      </Stack>
    </Box>
  )
}


export default function HomePage() {
  return (
    <>
      <Container maxWidth="lg" sx={{pt: 5}}>
        <Typography variant="h4">
          Welcome to fmr.ai!
        </Typography>

        <Typography sx={{mt: 3}}>
          This is a platform for visualizing and analyzing neural networks.
        </Typography>

        <PapersSection/>
      </Container>
    </>
  )
}