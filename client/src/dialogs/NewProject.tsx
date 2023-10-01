import {Dialog, DialogActions, DialogContent, DialogTitle, TextField} from "@mui/material";
import {useEffect, useState} from "react";
import {useMutation, useQueryClient} from "react-query";
import axios from "axios";
import {LoadingButton} from "@mui/lab";


interface NewProjectParams {
  name: string;
}

interface NewProjectResponse {
  uuid: string;
}

function useNewProjectMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (params: NewProjectParams) => {
      const response = await axios.post('/api/projects/new', params);
      return response.data as NewProjectResponse;
    },
    onSuccess() {
      queryClient.invalidateQueries('projects');
    }
  })
}


export interface NewProjectDialogProps {
  open: boolean;
  onClose: () => void;
}

export function NewProjectDialog(props: NewProjectDialogProps) {
  const [name, setName] = useState("");

  const newProject = useNewProjectMutation();

  useEffect(() => {
    if (props.open) {
      setName("");
    }
  }, [props.open]);

  const handleCreate = async () => {
    await newProject.mutateAsync({
      name,
    });

    props.onClose();
  };

  return (
    <Dialog
      open={props.open}
      onClose={props.onClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>New Project</DialogTitle>

      <DialogContent>
        <TextField
          fullWidth
          autoFocus
          autoComplete="off"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter project name"
        />
      </DialogContent>

      <DialogActions sx={{pr: 3, pb: 3}}>
        <LoadingButton
          variant="contained"
          onClick={handleCreate}
          loading={newProject.isLoading}
        >
          Create
        </LoadingButton>
      </DialogActions>
    </Dialog>
  );
}