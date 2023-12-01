import functools
import os.path

from fastapi import APIRouter, HTTPException, Depends

from fmrai.analysis.ffn_kv import KVRepository
from fmrai.analysis.ffn_kv.sqlite import SQLiteKVRepository

router = APIRouter(prefix='/api/kv')


@functools.cache
def get_kv_repo(workflow: str):
    db_path = f'./data/kv/{workflow}.sqlite'

    if '..' in workflow or '/' in workflow or '\\' in workflow or not os.path.isfile(db_path):
        raise HTTPException(400, f'Invalid workflow: {workflow}')

    return SQLiteKVRepository(db_path, readonly=True)


@router.get('/info')
def get_info(repo: KVRepository = Depends(get_kv_repo)):
    return {
        'stats': repo.stats()
    }


@router.get('/keys/heatmap')
def get_layer_key_heatmap(
        layer: int,
        sigma: float = 3.0,
        repo: KVRepository = Depends(get_kv_repo)
):
    return {
        'heatmap': repo.key_heatmap(layer, sigma),
    }


@router.get('/keys/neuron')
def get_key_neuron_details(
        layer: int,
        neuron: int,
        repo: KVRepository = Depends(get_kv_repo)
):
    return {
        'sentences': repo.key_sentences(layer, neuron),
    }
