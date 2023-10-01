import {Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, TextField} from "@mui/material";
import axios from "axios";
import {useMutation, useQueryClient} from "react-query";
import {Formik} from "formik";
import {LoadingButton} from "@mui/lab";


interface AddAgentFormValues {
  connect_url: string;
  agent_name: string;
  model_name: string;
}

interface AddAgentParams extends AddAgentFormValues {
  project_uuid: string;
}


interface AddAgentResponse {

}


function useAddAgentMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: AddAgentParams) => {
      const result = await axios.post('/api/projects/agents/add', params);
      return result.data as AddAgentResponse;
    },
    onSuccess() {
      queryClient.invalidateQueries('list-agents');
    }
  });
}


export interface NewAgentDialogProps {
  projectUuid?: string;
  open: boolean;
  onClose: () => void;
}


const INITIAL_VALUES = {
  connect_url: '',
  agent_name: '',
  model_name: '',
}


export function NewAgentDialog(props: NewAgentDialogProps) {
  const addAgent = useAddAgentMutation();

  const handleSubmit = async (values: AddAgentFormValues) => {
    if (!props.projectUuid)
      return;

    await addAgent.mutateAsync({
      ...values,
      project_uuid: props.projectUuid!,
    });

    props.onClose();
  };

  return (
    <Dialog
      open={Boolean(props.projectUuid && props.open)}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Add Agent</DialogTitle>

      <Formik
        initialValues={INITIAL_VALUES}
        onSubmit={handleSubmit}
      >
        {({getFieldProps, handleSubmit}) => (
          <>
            <DialogContent>
              <Stack spacing={2} sx={{ mt: 1 }}>
                <TextField
                  label="Connect URL"
                  placeholder="http://localhost:8001"
                  fullWidth
                  {...getFieldProps('connect_url')}
                />

                <TextField
                  label="Agent Name"
                  placeholder="default"
                  fullWidth
                  {...getFieldProps('agent_name')}
                />

                <TextField
                  label="Model Name"
                  placeholder="main"
                  fullWidth
                  {...getFieldProps('model_name')}
                />
              </Stack>
            </DialogContent>

            <DialogActions sx={{ pb: 3, pr: 3 }}>
              <LoadingButton
                variant="contained"
                sx={{ mr: 2 }}
                onClick={() => handleSubmit()}
                loading={addAgent.isLoading}
              >
                Add
              </LoadingButton>

              <Button onClick={() => props.onClose()}>
                Cancel
              </Button>
            </DialogActions>
          </>
        )}
      </Formik>
    </Dialog>
  )
}