from fmrai.agent.api import AgentAPI
from fmrai.agent.state import set_global_agent_state, AgentState, get_global_agent_state


class AgentServer:
    def __init__(self, api: AgentAPI):
        self.api = api

    def serve(self):
        from fmrai.agent.app import app
        import bottle

        assert get_global_agent_state() is None, 'Agent already running'

        set_global_agent_state(AgentState(
            api=self.api,
        ))

        bottle.run(app, host='localhost', port=8001)


def run_agent(api: AgentAPI):
    server = AgentServer(api)
    server.serve()
