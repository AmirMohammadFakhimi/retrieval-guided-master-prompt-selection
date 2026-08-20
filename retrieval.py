import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from lancedb.table import LanceTable

import embeddings
from dataset import Column

RETRIEVAL_METHODS = frozenset({'semantic', 'balanced_semantic'})
EXAMPLE_ORDERS = frozenset({
    'most_similar_first',
    'most_similar_last',
    'shuffle',
})
EXACT_RETRIEVAL_QUERY_BATCH_SIZE = 64


def build_training_filter(train: list[dict[str, Any]], train_size: int | None) -> str:
    """Describe the configured retrieval pool inside the complete master table."""

    if not train:
        raise ValueError('Cannot build a training filter from an empty pool')

    professions = sorted({str(row[Column.PROFESSION]) for row in train})
    quoted_professions = ', '.join(f"'{profession}'" for profession in professions)
    training_filter = f'{Column.PROFESSION.value} IN ({quoted_professions})'

    if train_size is not None:
        cutoff = max(int(row[Column.TRAIN_ORDER]) for row in train)
        training_filter += f' AND {Column.TRAIN_ORDER.value} <= {cutoff}'

    return training_filter


def _load_exact_training_matrix(
        training_table: LanceTable,
        training_filter: str,
        training_by_id: dict[str, dict[str, Any]],
        embedding_dimension: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Scan one filtered table in physical order and validate vector-row alignment."""

    if not training_by_id:
        raise ValueError('Cannot load an exact retrieval matrix from an empty training pool')
    if any(str(row[Column.ID]) != row_id for row_id, row in training_by_id.items()):
        raise ValueError('Training pool lookup keys do not align with row IDs')
    training_row_count = len(training_by_id)

    scanner = training_table.to_lance().scanner(
        columns=[
            Column.ID.value,
            Column.PROFESSION.value,
            Column.GENDER.value,
            Column.TRAIN_ORDER.value,
            'vector',
        ],
        filter=training_filter,
        scan_in_order=True,
    )
    vectors = np.empty((training_row_count, embedding_dimension), dtype=np.float32, order='C')
    ids: list[str] = []
    professions: list[str] = []
    genders: list[str] = []
    train_orders: list[int] = []
    row_offset = 0
    for batch in scanner.to_batches():
        batch_stop = row_offset + batch.num_rows
        if batch_stop > training_row_count:
            raise RuntimeError(
                f'Filtered embedding table contains more than the expected '
                f'{training_row_count} selected training rows'
            )
        vector_array = batch.column(batch.schema.get_field_index('vector'))
        vector_width = getattr(vector_array.type, 'list_size', None)
        if vector_width != embedding_dimension:
            raise RuntimeError(
                f'Embedding table vector width is {vector_width}; expected {embedding_dimension}'
            )
        if vector_array.null_count:
            raise RuntimeError('Exact training matrix contains null vectors')
        flat_vectors = vector_array.values.to_numpy(zero_copy_only=True)
        vectors[row_offset:batch_stop] = flat_vectors.reshape(batch.num_rows, embedding_dimension)
        ids.extend(str(value) for value in batch.column(batch.schema.get_field_index(Column.ID.value)).to_pylist())
        professions.extend(batch.column(batch.schema.get_field_index(Column.PROFESSION.value)).to_pylist())
        genders.extend(batch.column(batch.schema.get_field_index(Column.GENDER.value)).to_pylist())
        train_orders.extend(batch.column(batch.schema.get_field_index(Column.TRAIN_ORDER.value)).to_pylist())
        row_offset = batch_stop

    if row_offset != training_row_count:
        raise RuntimeError(
            f'Filtered embedding table contains {row_offset} rows; '
            f'expected {training_row_count} selected training rows'
        )
    if len(ids) != len(set(ids)):
        raise RuntimeError('Filtered embedding table contains duplicate IDs')
    if set(ids) != set(training_by_id):
        missing_ids = sorted(set(training_by_id) - set(ids))[:5]
        extra_ids = sorted(set(ids) - set(training_by_id))[:5]
        raise RuntimeError(
            f'Filtered embedding table IDs do not match the selected training pool; '
            f'missing={missing_ids}, extra={extra_ids}'
        )

    metadata_rows: list[dict[str, Any]] = []
    for row_id, profession, gender, train_order in zip(
            ids,
            professions,
            genders,
            train_orders,
            strict=True,
    ):
        source_row = training_by_id[row_id]
        actual_metadata = (profession, gender, train_order)
        expected_metadata = (
            source_row[Column.PROFESSION],
            source_row[Column.GENDER],
            source_row[Column.TRAIN_ORDER],
        )
        if actual_metadata != expected_metadata:
            raise RuntimeError(
                f'Filtered embedding metadata for {row_id!r} is misaligned: '
                f'{actual_metadata!r} != {expected_metadata!r}'
            )
        metadata_rows.append({
            Column.ID: row_id,
            Column.PROFESSION: profession,
            Column.GENDER: gender,
            Column.TRAIN_ORDER: train_order,
        })

    if not np.isfinite(vectors).all():
        raise RuntimeError('Exact training matrix contains non-finite values')
    training_norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(training_norms).all() or np.any(training_norms == 0):
        raise RuntimeError('Exact training matrix contains a zero vector')
    return metadata_rows, vectors, training_norms


def _rank_exact_cosine(
        training_vectors: np.ndarray,
        query_vectors: np.ndarray,
        training_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustively score and stably rank every training vector for every query."""

    query_norms = np.linalg.norm(query_vectors, axis=1)
    scores = query_vectors @ training_vectors.T
    scores /= query_norms[:, np.newaxis]
    scores /= training_norms[np.newaxis, :]
    if not np.isfinite(scores).all():
        raise RuntimeError('Exact cosine calculation produced non-finite scores')
    rankings = np.argsort(-scores, axis=1, kind='stable')
    return scores, rankings


def _validate_example_count(example_count: int, row_count: int) -> None:
    """Validate a positive retrieval size against the eligible pool."""

    if isinstance(example_count, bool) or not isinstance(example_count, int) or example_count < 1:
        raise ValueError('example_count must be a positive integer')
    if example_count > row_count:
        raise ValueError(f'Cannot retrieve {example_count} examples from {row_count} training rows')


def _select_balanced_candidates(
        ranking: np.ndarray,
        training_metadata: list[dict[str, Any]],
        example_count: int,
        profession_gender_pairs: tuple[tuple[str, str], ...],
) -> list[int]:
    """Greedily balance one complete semantic ranking without relaxing relevance."""

    if not profession_gender_pairs:
        raise ValueError('Balanced semantic retrieval requires profession-gender pairs')
    if ranking.ndim != 1 or len(ranking) != len(training_metadata):
        raise ValueError('Balanced selection requires one complete semantic ranking')

    professions = tuple(sorted({profession for profession, _ in profession_gender_pairs}))
    genders = tuple(sorted({gender for _, gender in profession_gender_pairs}))

    profession_counts = Counter()
    gender_counts = Counter()
    pair_counts = Counter()
    selected: list[int] = []
    selected_set: set[int] = set()

    while len(selected) < example_count:
        minimum_profession_count = min(profession_counts[profession] for profession in professions)
        minimum_gender_count = min(gender_counts[gender] for gender in genders)
        minimum_pair_count = min(pair_counts[pair] for pair in profession_gender_pairs)
        selected_index = next((
            int(index)
            for index in ranking
            if int(index) not in selected_set
            and profession_counts[training_metadata[int(index)][Column.PROFESSION]] == minimum_profession_count
            and gender_counts[training_metadata[int(index)][Column.GENDER]] == minimum_gender_count
            and pair_counts[(
                training_metadata[int(index)][Column.PROFESSION],
                training_metadata[int(index)][Column.GENDER],
            )] == minimum_pair_count
        ), None)
        if selected_index is None:
            break

        profession = training_metadata[selected_index][Column.PROFESSION]
        gender = training_metadata[selected_index][Column.GENDER]
        selected.append(selected_index)
        selected_set.add(selected_index)
        profession_counts[profession] += 1
        gender_counts[gender] += 1
        pair_counts[(profession, gender)] += 1

    if len(selected) != example_count:
        raise RuntimeError(
            f'Could not select {example_count} balanced semantic examples '
            f'from {len(training_metadata)} fully ranked training rows'
        )

    return selected


def _retrieve_exact_ids(
        training_metadata: list[dict[str, Any]],
        training_vectors: np.ndarray,
        training_norms: np.ndarray,
        query_ids: list[str],
        query_vectors: np.ndarray,
        retrieval_methods: list[str],
        max_example_count: int,
        profession_gender_pairs: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Compute exact maximum-k selections for all queries and requested methods."""

    unknown_methods = set(retrieval_methods) - RETRIEVAL_METHODS
    if unknown_methods:
        raise ValueError(f'Unknown retrieval methods: {sorted(unknown_methods)}')
    if len(retrieval_methods) != len(set(retrieval_methods)):
        raise ValueError('Retrieval methods cannot contain duplicates')
    if not retrieval_methods:
        raise ValueError('At least one retrieval method is required')
    _validate_example_count(max_example_count, len(training_metadata))
    if training_vectors.shape[0] != len(training_metadata):
        raise ValueError('Training vector rows do not align with training metadata')
    if query_vectors.shape[0] != len(query_ids):
        raise ValueError('Query vector rows do not align with query IDs')
    if len(query_ids) != len(set(query_ids)):
        raise ValueError('Query IDs must be unique')
    if 'balanced_semantic' in retrieval_methods:
        if not profession_gender_pairs:
            raise ValueError('Balanced semantic retrieval requires profession-gender pairs')
        metadata_pairs = {
            (row[Column.PROFESSION], row[Column.GENDER])
            for row in training_metadata
        }
        if not metadata_pairs.issubset(set(profession_gender_pairs)):
            raise ValueError('Balanced metadata contains a profession-gender pair outside the configured pool')
    selections: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for start in range(0, len(query_ids), EXACT_RETRIEVAL_QUERY_BATCH_SIZE):
        stop = min(start + EXACT_RETRIEVAL_QUERY_BATCH_SIZE, len(query_ids))
        scores, rankings = _rank_exact_cosine(
            training_vectors,
            query_vectors[start:stop],
            training_norms,
        )
        for block_index, query_id in enumerate(query_ids[start:stop]):
            ranking = rankings[block_index]
            for retrieval_method in retrieval_methods:
                if retrieval_method == 'semantic':
                    selected_indices = [int(index) for index in ranking[:max_example_count]]
                else:
                    selected_indices = _select_balanced_candidates(
                        ranking,
                        training_metadata,
                        max_example_count,
                        profession_gender_pairs,
                    )
                selections[(retrieval_method, query_id)] = [
                    (
                        str(training_metadata[index][Column.ID]),
                        float(scores[block_index, index]),
                    )
                    for index in selected_indices
                ]
        del scores, rankings
    return selections


def _hydrate_retrieval_selections(
        embedding_model_id: str,
        selections: dict[tuple[str, str], list[tuple[str, float]]],
        training_by_id: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Hydrate only selected IDs into the existing in-memory retrieval-cache shape."""

    hydrated: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (retrieval_method, query_id), selected in selections.items():
        hydrated[(retrieval_method, embedding_model_id, query_id)] = [
            {
                Column.ID: training_by_id[example_id][Column.ID],
                Column.SPLIT: training_by_id[example_id][Column.SPLIT],
                Column.HARD_TEXT: training_by_id[example_id][Column.HARD_TEXT],
                Column.PROFESSION: training_by_id[example_id][Column.PROFESSION],
                Column.GENDER: training_by_id[example_id][Column.GENDER],
                'retrieval_score': score,
            }
            for example_id, score in selected
        ]
    return hydrated


def prepare_exact_retrievals(
        training_table: LanceTable,
        training_by_id: dict[str, dict[str, Any]],
        training_filter: str,
        evaluation_rows: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        retrieval_methods: list[str],
        max_example_count: int,
        profession_gender_pairs: tuple[tuple[str, str], ...],
        device: str,
        runtime_cache_path: Path,
        progress: Callable[[str], None] = print,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Compute configured exact retrievals once before language-model loading."""

    query_ids = [str(row[Column.ID]) for row in evaluation_rows]
    query_vectors = embeddings.load_or_encode_embedding_queries(
        evaluation_rows,
        embedding_model,
        device,
        runtime_cache_path,
    )

    progress(
        f'Scanning {len(training_by_id)} eligible vectors once for exact exhaustive '
        f'{embedding_model["id"]} retrieval'
    )
    training_metadata, training_vectors, training_norms = _load_exact_training_matrix(
        training_table,
        training_filter,
        training_by_id,
        embedding_model['dimension'],
    )
    try:
        selections = _retrieve_exact_ids(
            training_metadata,
            training_vectors,
            training_norms,
            query_ids,
            query_vectors,
            retrieval_methods,
            max_example_count,
            profession_gender_pairs,
        )
    finally:
        del training_metadata, training_vectors, training_norms, query_vectors

    return _hydrate_retrieval_selections(
        embedding_model['id'],
        selections,
        training_by_id,
    )


def order_examples(
        examples: list[dict[str, Any]],
        order: str,
        seed: int,
) -> list[dict[str, Any]]:
    """Order one already-selected demonstration set for prompt presentation."""

    if order not in EXAMPLE_ORDERS:
        raise ValueError(f'Unknown example order {order!r}; expected one of {sorted(EXAMPLE_ORDERS)}')

    if order == 'most_similar_first':
        return sorted(
            examples,
            key=lambda example: example['retrieval_score'],
            reverse=True,
        )

    if order == 'most_similar_last':
        return sorted(
            examples,
            key=lambda example: example['retrieval_score'],
        )

    if order == 'shuffle':
        random.Random(seed).shuffle(examples)
        return examples

    raise RuntimeError(f'Example order {order!r} is allowed but not implemented')
