import {atom, useAtomValue} from "jotai";
import {Agent} from "../api/types.ts";


export const currentModelAtom = atom<null | string>(null);

export const agentListAtom = atom<null | Agent[]>(null);



export function useAgentByModelName(name: null | string): Agent | undefined {
  const agents = useAtomValue(agentListAtom);

  if (!name)
    return undefined;

  return agents?.find(agent => agent.model_name === name);
}
