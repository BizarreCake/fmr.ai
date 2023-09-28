import {Agent, Project} from "./types.ts";
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


export interface ListAgentsParams {
  project_uuid: string;
}

export interface ListAgentsResponse {
  agents: Agent[];
}

export function useListAgentsQuery(params?: ListAgentsParams) {
  return useQuery(['list-agents', params], async () => {
    const result = await axios.get('/api/projects/agents/list', {
      params,
    });
    return result.data as ListAgentsResponse;
  }, {
    enabled: !!params?.project_uuid,
  });
}
