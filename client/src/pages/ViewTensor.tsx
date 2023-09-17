import {Box, Card, IconButton, Stack, Typography} from "@mui/material";
import {useNavigate, useParams} from "react-router";
import {ArrowBack} from "@mui/icons-material";
import {useQuery} from "react-query";
import axios from "axios";


interface GetTensorInfoParams {
  tensor_id?: string;
  time_step: number;
}

interface TensorInfo {
  format: string;
}

interface GetTensorInfoResponse {
  tensor: null | TensorInfo;
}


function useGetTensorInfoQuery(params: GetTensorInfoParams) {
  return useQuery('tensor-info', async () => {
    const result = await axios.get(
      '/api/tensor/info',
      { params, }
    );
    return result.data as GetTensorInfoResponse;
  }, {
    enabled: Boolean(params.tensor_id),
  });
}


export default function ViewTensorPage() {
  const { tensorId } = useParams();
  const { data } = useGetTensorInfoQuery({
    tensor_id: tensorId,
    time_step: 0,
  });

  const navigate = useNavigate();
  const handleBack = () => {
    navigate(-1);
  };

  return (
    <Box sx={{ p: 3 }}>

      <Stack direction="row" alignItems="center" spacing={1}>
        <IconButton onClick={handleBack}>
          <ArrowBack />
        </IconButton>

        <Typography variant="h6">
          Tensor <span style={{ fontFamily: 'monospace' }}>{tensorId}</span>
        </Typography>
      </Stack>

      <Stack alignItems="center">
        {data?.tensor?.format === 'image' && (
          <Card sx={{ p: 3, mt: 3 }}>
            <img
              alt="Tensor image"
              src={`/api/tensor/image?tensor_id=${tensorId}&time_step=0`}
            />
          </Card>
        )}
      </Stack>

    </Box>
  );
}
