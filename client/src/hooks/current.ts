import {useParams} from "react-router";
import {useGetProjectQuery} from "../api/routes.ts";


export function useCurrentProjectQuery() {
  const { projectId } = useParams();
  return useGetProjectQuery(projectId ? {uuid: projectId} : undefined);
}
