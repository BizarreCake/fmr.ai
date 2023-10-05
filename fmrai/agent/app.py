import bottle

from fmrai.agent.logic import do_list_attention_head_plot_inputs, do_find_attention, do_extract_attention, \
    do_get_model_graph
from fmrai.agent.logic import do_predict_text, do_generate_model_graph, do_compute_attention_head_plot
from fmrai.agent.state import get_global_agent_state

app = bottle.Bottle()


@app.get('/status')
def get_status():
    return {
        'status': repr(get_global_agent_state())
    }


@app.post('/model/graph/generate')
def generate_model_graph():
    data = bottle.request.json
    root_dir = data['root_dir']
    model_name = data['model_name']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    do_generate_model_graph(agent_state, root_dir=root_dir, model_name=model_name)


@app.post('/model/graph/get')
def get_model_graph():
    data = bottle.request.json
    root_dir = data['root_dir']
    model_name = data['model_name']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    return do_get_model_graph(agent_state, root_dir=root_dir, model_name=model_name)


@app.post('/model/find_attention')
def find_attention():
    data = bottle.request.json
    root_dir = data['root_dir']
    model_name = data['model_name']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    return do_find_attention(agent_state, root_dir=root_dir, model_name=model_name).dict()


@app.post('/analyze/text/extract_attention')
def extract_attention():
    data = bottle.request.json
    key = data['key']
    tensor_id = data['tensor_id']
    root_dir = data['root_dir']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    return do_extract_attention(agent_state, key, tensor_id, root_dir=root_dir).dict()


@app.post('/predict/text/one')
def predict_text_one():
    data = bottle.request.json
    text = data['text']
    root_dir = data['root_dir']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    result = do_predict_text(agent_state, text, root_dir=root_dir)
    print('predict:', result)

    return {
        'activation_map_key': result.activation_map_key,
        'token_ids': result.result.token_ids,
        'token_names': result.result.token_names,
    }


@app.get('/datasets/list')
def list_datasets():
    agent_state = get_global_agent_state()
    assert agent_state.api is not None
    return agent_state.api.list_datasets().model_dump()


@app.post('/analyze/attention/head_plot/compute')
def compute_attention_head_plot():
    data = bottle.request.json
    dataset_name = data['dataset']
    limit = data['limit']
    root_dir = data['root_dir']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    return do_compute_attention_head_plot(
        agent_state, dataset_name, limit, root_dir=root_dir
    )


@app.post('/analyze/attention/head_plot/list_inputs')
def list_attention_head_plot_inputs():
    data = bottle.request.json
    key = data['key']
    limit = data.get('limit', 100)
    root_dir = data['root_dir']

    agent_state = get_global_agent_state()
    assert agent_state.api is not None

    return do_list_attention_head_plot_inputs(
        agent_state, key, limit, root_dir=root_dir
    )
