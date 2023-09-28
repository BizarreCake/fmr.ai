from fastapi import APIRouter

from server.entrypoints.web import models

router = APIRouter(prefix='/api')


@router.post('/model-graph/generate')
def generate_model_graph(data: models.GenerateModelGraphIn):
    print(data)
