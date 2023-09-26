import {Project} from "./types.ts";
import {useQuery} from "react-query";
import axios from "axios";


export interface GetProjectParams {
  uuid: string;
}

export interface GetProjectResponse {
  project: null | Project;
}

export function useGetProjectQuery(params?: GetProjectParams) {
  return useQuery(['project', params?.uuid], async () => {
    const response = await axios.get(`/api/projects/get`, {params,});
    return response.data as GetProjectResponse;
  }, {
    enabled: !!params?.uuid,
  });
}
