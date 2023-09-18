import bottle

from fmrai.agent.logic import do_predict_text, do_init_model
from fmrai.agent.state import get_global_agent_state

app = bottle.Bottle()


@app.get('/status')
def get_status():
    return {
        'status': repr(get_global_agent_state())
    }


@app.post('/model/init')
def init_model():
    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    do_init_model(agent_state)


@app.post('/predict/text/one')
def predict_text_one():
    data = bottle.request.json
    text = data['text']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    result = do_predict_text(agent_state, text)
    print('predict:', result)

    return {
        'activation_map_key': result.activation_map_key,
        'token_ids': result.result.token_ids,
        'token_names': result.result.token_names,
    }


@app.post('/predict/text/many')
def predict_text_many():
    pass
