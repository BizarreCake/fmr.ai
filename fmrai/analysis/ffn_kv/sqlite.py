import json
from typing import List, Tuple, Dict, Optional

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.sql.expression import delete, select, distinct, insert
from sqlalchemy.sql import func

from fmrai.analysis.common import DatapointTokenization
from fmrai.analysis.ffn_kv import KVStats
from fmrai.analysis.ffn_kv.base import KVRepository, KVResults, NeuronKVRecord, KVKeyHeatmap, KVKeySentence

Base = declarative_base()


class Tokenization(Base):
    __tablename__ = 'tokenization'

    id = Column(Integer, primary_key=True)
    words = Column(String)


class NeuronBestCandidate(Base):
    __tablename__ = 'neuron_best_candidate'

    id = Column(Integer, primary_key=True, autoincrement=True)

    layer = Column(Integer, nullable=False, index=True)
    neuron = Column(Integer, nullable=False, index=True)

    sentence_id = Column(Integer, nullable=False)
    prefix_len = Column(Integer)

    value = Column(Float)
    z = Column(Float)
    z_abs = Column(Float, index=True)


class SQLiteKVRepository(KVRepository):
    def __init__(self, db_path: str, max_candidates: int = 10, *, readonly=True):
        import sqlalchemy.orm

        self._max_candidates = max_candidates

        self._engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        self._session = sqlalchemy.orm.sessionmaker(bind=self._engine)

        if not readonly:
            self._init()

    def _init(self):
        # create tables
        Base.metadata.create_all(self._engine)

    def _bulk_add_records(self, results: KVResults, session):
        records = []
        for layer, neuron in results.keys():
            for record in results.get(layer, neuron):
                records.append(NeuronBestCandidate(
                    layer=layer,
                    neuron=neuron,
                    sentence_id=record.instance_idx,
                    prefix_len=record.prefix_idx,
                    value=record.value,
                    z=record.z,
                    z_abs=abs(record.z),
                ))

        session.bulk_save_objects(records)

    def _trim_candidates(self, n: int, session):
        """
        Delete all but the best N candidate for each neuron.
        """

        subq = select(
            func.row_number().over(
                partition_by=[NeuronBestCandidate.layer, NeuronBestCandidate.neuron],
                order_by=NeuronBestCandidate.z_abs.desc(),
            ).label('rn'),
            NeuronBestCandidate.id.label('id'),
        ).alias('subq')

        q = delete(NeuronBestCandidate).where(
            NeuronBestCandidate.id.in_(
                select(subq.c.id).where(subq.c.rn > n)
            )
        )

        session.execute(q)

    def _save_tokenization(self, tokenization: Dict[int, DatapointTokenization], session):
        stmt = (
            insert(Tokenization.__table__)
            .values([
                {
                    'id': sentence_id,
                    'words': json.dumps(tokenization.words),
                }
                for sentence_id, tokenization in tokenization.items()
            ])
            .prefix_with('OR REPLACE')
        )

        session.execute(stmt)

    def add(self, results: KVResults):
        with self._session() as session:
            self._bulk_add_records(results, session)
            session.commit()

            self._trim_candidates(self._max_candidates, session)
            session.commit()

            if results.tokenization is not None:
                self._save_tokenization(results.tokenization, session)
                session.commit()

    def stats(self) -> KVStats:
        with self._session() as session:
            # count number of layers
            num_layers = session.scalar(
                select(func.count())
                .select_from(
                    select(NeuronBestCandidate.layer)
                    .distinct()
                    .subquery()
                )
            )

            # count number of neurons per layer
            # noinspection PyTypeChecker
            num_neurons_per_layer = dict(list(session.execute(
                select(
                    NeuronBestCandidate.layer,
                    func.count(distinct(NeuronBestCandidate.neuron))
                )
                .group_by(NeuronBestCandidate.layer)
            )))

            return KVStats(
                num_layers=num_layers,
                num_key_neurons_per_layer=num_neurons_per_layer,
            )

    def key_heatmap(self, layer: int, sigma: float = 3.0) -> KVKeyHeatmap:
        with self._session() as session:
            # count number of neurons in layer
            num_key_neurons = session.scalar(
                select(
                    func.count(distinct(NeuronBestCandidate.neuron))
                )
                .where(
                    NeuronBestCandidate.layer == layer,
                )
            )

            results = list(session.execute(
                select(
                    NeuronBestCandidate.neuron,
                    func.count(
                        NeuronBestCandidate.sentence_id
                    )
                )
                .where(
                    NeuronBestCandidate.layer == layer,
                    NeuronBestCandidate.z_abs > sigma,
                )
                .group_by(
                    NeuronBestCandidate.neuron,
                )
                .order_by(
                    NeuronBestCandidate.neuron,
                )
            ))

            # construct heatmap
            heatmap = [0] * num_key_neurons
            for neuron, count in results:
                heatmap[neuron] = count

            return KVKeyHeatmap(
                layer=layer,
                sigma=sigma,
                num_neurons=num_key_neurons,
                heatmap=heatmap,
            )

    def key_sentences(self, layer: int, neuron: int, limit: Optional[int] = None) -> List[KVKeySentence]:
        with self._session() as session:
            # get sentence ids
            results = list(session.execute(
                select(
                    NeuronBestCandidate.sentence_id,
                    NeuronBestCandidate.prefix_len,
                    NeuronBestCandidate.value,
                    NeuronBestCandidate.z,
                    Tokenization.words,
                )
                .outerjoin(Tokenization, Tokenization.id == NeuronBestCandidate.sentence_id)
                .where(
                    NeuronBestCandidate.layer == layer,
                    NeuronBestCandidate.neuron == neuron,
                )
                .order_by(
                    NeuronBestCandidate.z_abs.desc(),
                )
                .limit(limit)
            ))

            return [
                KVKeySentence(
                    sentence_id=sentence_id,
                    prefix_len=prefix_len,
                    value=value,
                    z=z,
                    words=json.loads(words) if words is not None else None,
                )
                for sentence_id, prefix_len, value, z, words in results
            ]
