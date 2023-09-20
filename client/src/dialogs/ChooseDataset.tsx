import {
  Alert,
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  LinearProgress,
  List,
  ListItemButton,
  Typography
} from "@mui/material";
import axios from "axios";
import {useQuery} from "react-query";
import { Dataset } from "../api/types";



interface ListDatasetsResponse {
  datasets: Dataset[];
}

function useListDatasetsQuery() {
  return useQuery('list-datasets', async () => {
    const result = await axios.get(
      '/api/datasets/list',
    );
    return result.data as ListDatasetsResponse;
  });
}


export interface ChooseDatasetDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (dataset: Dataset, limit?: number) => void;
}


export function ChooseDatasetDialog(props: ChooseDatasetDialogProps) {
  const {data, isLoading} = useListDatasetsQuery();

  const handleClickDataset = (dataset: Dataset) => {
    props.onSelect(dataset);
  }

  return (
    <Dialog
      open={props.open}
      // onClose={(data?.datasets && data.datasets.length === 0) ? props.onClose : undefined}
      onClose={props.onClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>Choose Dataset</DialogTitle>

      <DialogContent>
        {isLoading && <LinearProgress/>}

        {!isLoading && data && (
          <>
            <List>
              {data.datasets.map((dataset, i) => (
                <ListItemButton
                  key={i}
                  onClick={() => handleClickDataset(dataset)}
                >
                  <Box>
                    <Typography variant="subtitle2">{dataset.name}</Typography>
                    <Typography
                      variant="body2"
                      fontStyle={dataset.description ? 'normal' : 'italic'}
                      color={dataset.description ? 'text.primary' : 'text.secondary'}
                    >
                      {dataset.description ?? 'No description available.'}
                    </Typography>
                  </Box>
                </ListItemButton>
              ))}
            </List>
          </>
        )}

        {!isLoading && data?.datasets && data.datasets.length === 0 && (
          <>
            <Alert severity="warning">
              No datasets available.
            </Alert>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}