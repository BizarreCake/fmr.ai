import {useEffect} from "react";
import {useNavigate, useParams} from "react-router";


export default function ProjectHomePage() {
  const { projectId } = useParams();

  const navigate = useNavigate();
  useEffect(() => {
    if (projectId)
      navigate(`/project/${projectId}/agents`);
  }, [projectId, navigate]);

  return <></>;
}