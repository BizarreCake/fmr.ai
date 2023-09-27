import {Button, Dialog, DialogActions, DialogContent, DialogTitle, Stack, TextField} from "@mui/material";


export interface NewAgentDialogProps {
  open: boolean;
  onClose: () => void;
}

export function NewAgentDialog(props: NewAgentDialogProps) {
  return (
    <Dialog
      open={props.open}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Add Agent</DialogTitle>

      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <TextField
            label="Connect URL"
            placeholder="http://localhost:8001"
            fullWidth
          />

          <TextField
            label="Agent Name"
            placeholder="Default Agent"
            fullWidth
          />
        </Stack>
      </DialogContent>

      <DialogActions sx={{ pb: 3, pr: 3 }}>
        <Button variant="contained" sx={{ mr: 2 }}>
          Add
        </Button>

        <Button onClick={() => props.onClose()}>
          Cancel
        </Button>
      </DialogActions>
    </Dialog>
  )
}