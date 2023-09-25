import {Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField} from "@mui/material";
import {useEffect, useState} from "react";
import {useMutation} from "react-query";
import axios from "axios";


interface NewProjectParams {
  name: string;
}

interface NewProjectResponse {

}

function useNewProjectMutation() {
  return useMutation({
    mutationFn: async (params: NewProjectParams) => {
      const response = await axios.post('/api/projects/new', params);
      return response.data as NewProjectResponse;
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
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter project name"
        />
      </DialogContent>

      <DialogActions sx={{ pr: 3, pb: 3 }}>
        <Button variant="contained">
          Create
        </Button>
      </DialogActions>
    </Dialog>
  )
}