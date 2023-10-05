from fmrai.agent.api import AgentAPI
from fmrai.agent.state import set_global_agent_state, AgentState, get_global_agent_state


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8001


class AgentServer:
    def __init__(
            self,
            api: AgentAPI,
            host=None,
            port=None,
    ):
        self.api = api
        self.host = host or DEFAULT_HOST
        self.port = port or DEFAULT_PORT

    def serve(self):
        from fmrai.agent.app import app
        import bottle

        assert get_global_agent_state() is None, 'Agent already running'

        set_global_agent_state(AgentState(
            api=self.api,
        ))

        bottle.run(app, host=self.host, port=self.port)


def run_agent(api: AgentAPI, host=None, port=None):
    server = AgentServer(api, host=host, port=port)
    server.serve()
