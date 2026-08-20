import hashlib
import json
import os
import tempfile
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np
from lancedb.table import LanceTable
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from dataset import Column
from modeling import TORCH_DTYPES, clear_model_memory

LANCEDB_INGEST_BATCH_SIZE = 2048
TRAINING_TABLE_VERSION = 1
QUERY_VECTOR_CACHE_VERSION = 1


def _warn_about_embedding_input_truncation(
        encoder: SentenceTransformer,
        rows: list[dict[str, Any]],
        embedding_model_id: str,
        max_sequence_length: int,
        input_kind: str,
        prompt: str = '',
) -> None:
    """Report embedding inputs that SentenceTransformers will truncate."""

    if not rows:
        return

    texts = [f'{prompt}{row[Column.HARD_TEXT]}' for row in rows]
    tokenized = encoder.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        verbose=False,
    )

    overlength_inputs = []
    for row, token_ids in zip(rows, tokenized['input_ids'], strict=True):
        token_count = len(token_ids)
        if token_count > max_sequence_length:
            overlength_inputs.append(f'{row[Column.ID]!r} ({token_count} tokens)')

    if overlength_inputs:
        warnings.warn(
            f'{embedding_model_id} will truncate {len(overlength_inputs)} {input_kind} input(s) '
            f'to max_sequence_length={max_sequence_length}: {", ".join(overlength_inputs)}',
            stacklevel=2,
        )


def _embedding_table_name(embedding_model_id: str) -> str:
    """Return the stable LanceDB table name for an embedding model."""

    return f'semantic_{embedding_model_id.lower().replace("/", "_")}'


def _load_embedding_encoder(embedding_model: dict[str, Any], device: str) -> SentenceTransformer:
    """Load one pinned embedding model and apply its configured sequence limit."""

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    dtype_name = embedding_model['dtype']

    encoder = SentenceTransformer(
        embedding_model_id,
        revision=embedding_model['revision'],
        device=device,
        model_kwargs={'dtype': TORCH_DTYPES[dtype_name]},
        truncate_dim=embedding_dimension,
    )
    native_max_sequence_length = cast(int, encoder.get_max_seq_length())
    if max_sequence_length > native_max_sequence_length:
        raise ValueError(
            f'{embedding_model_id} supports at most {native_max_sequence_length} tokens, '
            f'but max_sequence_length={max_sequence_length} was configured'
        )
    encoder.max_seq_length = max_sequence_length
    return encoder


def _valid_embedding_vectors(vectors: np.ndarray) -> bool:
    """Return whether every vector and its norm are finite and nonzero."""

    with np.errstate(over='ignore', invalid='ignore'):
        norms = np.linalg.norm(vectors, axis=1)
    return bool(
        np.isfinite(vectors).all()
        and np.isfinite(norms).all()
        and np.all(norms != 0)
    )


def _canonical_json(value: Any) -> str:
    """Serialize fingerprint inputs independently of dictionary insertion order."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':'),
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    """Return the SHA-256 digest of one canonical JSON value."""

    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _fingerprint_row_bytes(row: dict[str, Any]) -> bytes:
    """Encode one source row for ordered content fingerprints."""

    return _canonical_json([
        row[Column.ID],
        row[Column.SPLIT],
        row[Column.HARD_TEXT],
        row[Column.PROFESSION],
        row[Column.GENDER],
        row.get(Column.TRAIN_ORDER),
    ]).encode('utf-8')


def fingerprint_rows(rows: list[dict[str, Any]]) -> str:
    """Hash canonical row contents in their exact supplied order."""

    digest = hashlib.sha256()
    digest.update(b'[')
    for index, row in enumerate(rows):
        if index:
            digest.update(b',')
        digest.update(_fingerprint_row_bytes(row))
    digest.update(b']')
    return digest.hexdigest()


def _embedding_model_slug(embedding_model_id: str) -> str:
    """Return one filesystem-safe embedding-model ID."""

    return embedding_model_id.lower().replace('/', '_')


def _embedding_fingerprint_settings(embedding_model: dict[str, Any]) -> dict[str, Any]:
    """Return only settings that can change encoded vector values."""

    return {
        'model_id': embedding_model['id'],
        'revision': embedding_model['revision'],
        'dimension': embedding_model['dimension'],
        'max_sequence_length': embedding_model['max_sequence_length'],
        'dtype': embedding_model['dtype'],
        'normalize_embeddings': True,
    }


def _training_table_manifest(
        training_rows_digest: str,
        training_row_count: int,
        embedding_model: dict[str, Any],
) -> dict[str, Any]:
    """Describe every input needed to trust one complete training table."""

    return {
        'version': TRAINING_TABLE_VERSION,
        'training_rows_digest': training_rows_digest,
        'training_row_count': training_row_count,
        'physical_row_order': 'stable_descending_hard_text_character_length',
        'embedding': _embedding_fingerprint_settings(embedding_model),
    }


def _query_vector_cache_key(
        queries: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
) -> str:
    """Identify cached query vectors by exactly the inputs that affect them."""

    return _fingerprint({
        'version': QUERY_VECTOR_CACHE_VERSION,
        'queries': [
            [row[Column.ID], row[Column.HARD_TEXT]]
            for row in queries
        ],
        'query_prompt': embedding_model['query_prompt'],
        'resolved_device': device,
        'embedding': _embedding_fingerprint_settings(embedding_model),
    })


def _atomic_write_json(path: Path, value: Any) -> None:
    """Publish one JSON file without exposing an incomplete destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=path.parent,
                prefix=f'.{path.name}.',
                suffix='.tmp',
                delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        os.replace(temporary_path, path)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Publish one pickle-free NPZ cache atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=path.parent,
                prefix=f'.{path.name}.',
                suffix='.tmp',
                delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez(handle, **arrays)
        os.replace(temporary_path, path)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any | None:
    """Return decoded JSON, or None when a file is absent or malformed."""

    try:
        with path.open(encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError, UnicodeError):
        return None


def _has_matching_manifest(path: Path, expected_manifest: dict[str, Any]) -> bool:
    """Compare a readable manifest without Python's cross-type numeric equality."""

    try:
        return _canonical_json(_load_json(path)) == _canonical_json(expected_manifest)
    except (TypeError, ValueError):
        return False


def _training_manifest_path(database_path: Path, table_name: str) -> Path:
    """Return the external manifest path for one LanceDB table."""

    return database_path / '_manifests' / f'{table_name}.json'


def _valid_training_table_schema(
        table: LanceTable,
        embedding_dimension: int,
        expected_row_count: int,
) -> bool:
    """Check the complete table's row count and exact persisted schema."""

    expected_columns = [
        Column.ID,
        Column.PROFESSION,
        Column.GENDER,
        Column.TRAIN_ORDER,
        'vector',
    ]
    schema = table.schema
    vector_type = schema.field('vector').type if 'vector' in schema.names else None
    return (
        table.count_rows() == expected_row_count
        and schema.names == expected_columns
        and getattr(vector_type, 'list_size', None) == embedding_dimension
        and str(getattr(vector_type, 'value_type', '')) == 'float'
    )


def prepare_training_embedding_table(
        training_rows: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
        database_path: Path,
        training_rows_digest: str,
        progress: Callable[[str], None] = print,
) -> LanceTable:
    """Explicitly build or reuse all training embeddings for one model."""

    if not training_rows:
        raise ValueError('Cannot prepare an embedding table from an empty training corpus')

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    batch_size = embedding_model['batch_size']
    table_name = _embedding_table_name(embedding_model_id)
    manifest = _training_table_manifest(training_rows_digest, len(training_rows), embedding_model)
    manifest_path = _training_manifest_path(database_path, table_name)

    database_path.mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(database_path)
    if table_name in database.list_tables().tables:
        table: LanceTable = cast(LanceTable, database.open_table(table_name))
        table_is_valid = (
            _has_matching_manifest(manifest_path, manifest)
            and _valid_training_table_schema(table, embedding_dimension, len(training_rows))
        )
        if table_is_valid:
            progress(f'Reusing {len(training_rows)} manifested training embeddings from {database_path / table_name}')
            return table

        progress(f'Rebuilding incompatible or unmanifested LanceDB table {table_name}')
        database.drop_table(table_name)
        manifest_path.unlink(missing_ok=True)

    length_sorted_training_rows = sorted(
        training_rows,
        key=lambda row: len(row[Column.HARD_TEXT]),
        reverse=True,
    )
    progress(f'Embedding all {len(length_sorted_training_rows)} training rows for {embedding_model_id}')
    encoder: SentenceTransformer | None = None
    table: LanceTable | None = None

    try:
        encoder: SentenceTransformer = _load_embedding_encoder(embedding_model, device)
        progress_bar = tqdm(
            total=len(length_sorted_training_rows),
            desc='Embedding all training rows for LanceDB',
            unit='row',
            leave=True,
        )
        try:
            for start in range(0, len(length_sorted_training_rows), LANCEDB_INGEST_BATCH_SIZE):
                batch = length_sorted_training_rows[start:start + LANCEDB_INGEST_BATCH_SIZE]
                _warn_about_embedding_input_truncation(
                    encoder,
                    batch,
                    embedding_model_id,
                    max_sequence_length,
                    'training-document',
                )
                # SentenceTransformer re-sorts this bounded input internally
                # and restores its returned vectors to the batch's input order.
                vectors = np.asarray(
                    encoder.encode(
                        [row[Column.HARD_TEXT] for row in batch],
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )

                if vectors.ndim != 2 or vectors.shape != (len(batch), embedding_dimension):
                    raise RuntimeError(
                        f'Embedding model returned vectors with shape '
                        f'{vectors.shape}; expected ({len(batch)}, {embedding_dimension})'
                    )
                if not _valid_embedding_vectors(vectors):
                    raise RuntimeError('Embedding model returned a non-finite or zero training vector')

                records = [
                    {
                        Column.ID: row[Column.ID],
                        Column.PROFESSION: row[Column.PROFESSION],
                        Column.GENDER: row[Column.GENDER],
                        Column.TRAIN_ORDER: row[Column.TRAIN_ORDER],
                        'vector': vector.tolist(),
                    }
                    for row, vector in zip(batch, vectors, strict=True)
                ]

                if table is None:
                    table = cast(LanceTable, database.create_table(table_name, data=records))
                else:
                    table.add(records)

                progress_bar.update(len(batch))
        except BaseException:
            if table_name in database.list_tables().tables:
                database.drop_table(table_name)
            manifest_path.unlink(missing_ok=True)
            raise
        finally:
            if progress_bar is not None:
                progress_bar.close()
    finally:
        if encoder is not None:
            del encoder
        clear_model_memory(device)

    if table is None:
        raise RuntimeError('Cannot build a LanceDB table from an empty training corpus')
    try:
        progress(f'Compacting new LanceDB table {table_name}')
        table.optimize()
        _atomic_write_json(manifest_path, manifest)
    except BaseException:
        if table_name in database.list_tables().tables:
            database.drop_table(table_name)
        manifest_path.unlink(missing_ok=True)
        raise
    return table


def open_training_embedding_table(
        training_rows: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        database_path: Path,
        training_rows_digest: str,
) -> LanceTable:
    """Open a table only when it contains every canonical training row."""

    if not database_path.exists():
        raise RuntimeError(
            f'Training-embedding table for {embedding_model["id"]} is missing. '
            f'Run "Prepare all training embeddings" before the experiment.'
        )

    database = lancedb.connect(database_path)
    table_name = _embedding_table_name(embedding_model['id'])
    if table_name not in database.list_tables().tables:
        raise RuntimeError(
            f'Training-embedding table for {embedding_model["id"]} is missing. '
            f'Run "Prepare all training embeddings" before the experiment.'
        )

    table = cast(LanceTable, database.open_table(table_name))
    manifest = _training_table_manifest(
        training_rows_digest,
        len(training_rows),
        embedding_model,
    )
    manifest_path = _training_manifest_path(database_path, table_name)
    if (
            not _has_matching_manifest(manifest_path, manifest)
            or not _valid_training_table_schema(
                table,
                embedding_model['dimension'],
                len(training_rows),
            )
    ):
        raise RuntimeError(
            f'Training-embedding table for {embedding_model["id"]} is unmanifested or incompatible. '
            f'Run "Prepare all training embeddings" before the experiment.'
        )

    return table


def _encode_embedding_queries(
        queries: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
) -> np.ndarray:
    """Encode only the selected evaluation queries for one cached model."""

    embedding_model_id = embedding_model['id']
    embedding_dimension = embedding_model['dimension']
    max_sequence_length = embedding_model['max_sequence_length']
    batch_size = embedding_model['batch_size']
    query_prompt = embedding_model['query_prompt']
    encoder: SentenceTransformer | None = None

    try:
        encoder: SentenceTransformer = _load_embedding_encoder(embedding_model, device)
        _warn_about_embedding_input_truncation(
            encoder,
            queries,
            embedding_model_id,
            max_sequence_length,
            'query',
            query_prompt,
        )
        query_vectors = np.asarray(
            encoder.encode(
                [row[Column.HARD_TEXT] for row in queries],
                prompt=query_prompt,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
    finally:
        if encoder is not None:
            del encoder
        clear_model_memory(device)

    if query_vectors.ndim != 2 or query_vectors.shape != (len(queries), embedding_dimension):
        raise RuntimeError(
            f'Embedding model returned query vectors with shape '
            f'{query_vectors.shape}; expected '
            f'({len(queries)}, {embedding_dimension})'
        )
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if not _valid_embedding_vectors(query_vectors):
        raise RuntimeError('Embedding model returned a non-finite or zero query vector')
    return query_vectors


def _query_vector_cache_path(
        runtime_cache_path: Path,
        embedding_model_id: str,
        cache_key: str,
) -> Path:
    """Return the fingerprinted NPZ path for evaluation vectors."""

    return runtime_cache_path / 'query_vectors' / _embedding_model_slug(embedding_model_id) / f'{cache_key}.npz'


def _load_query_vector_cache(
        path: Path,
        expected_query_ids: list[str],
        embedding_dimension: int,
) -> np.ndarray | None:
    """Load and strictly validate one pickle-free query-vector artifact."""

    try:
        with path.open('rb') as handle:
            with np.load(handle, allow_pickle=False) as payload:
                if set(payload.files) != {'query_ids', 'vectors'}:
                    return None
                query_ids = np.asarray(payload['query_ids'])
                vectors = np.asarray(payload['vectors'])
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile):
        return None

    if (
            query_ids.ndim != 1
            or query_ids.dtype.kind not in {'U', 'S'}
            or [str(query_id) for query_id in query_ids.tolist()] != expected_query_ids
            or vectors.dtype != np.float32
            or vectors.shape != (len(expected_query_ids), embedding_dimension)
            or not vectors.flags.c_contiguous
    ):
        return None
    if not _valid_embedding_vectors(vectors):
        return None
    return vectors


def load_or_encode_embedding_queries(
        queries: list[dict[str, Any]],
        embedding_model: dict[str, Any],
        device: str,
        runtime_cache_path: Path,
) -> np.ndarray:
    """Reuse or atomically cache all selected evaluation-query vectors."""

    query_ids = [str(row[Column.ID]) for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError('Evaluation query IDs must be unique')

    query_cache_key = _query_vector_cache_key(queries, embedding_model, device)
    path = _query_vector_cache_path(runtime_cache_path, embedding_model['id'], query_cache_key)
    cached_vectors = _load_query_vector_cache(
        path,
        query_ids,
        embedding_model['dimension'],
    )
    if cached_vectors is not None:
        return cached_vectors

    vectors = _encode_embedding_queries(queries, embedding_model, device)
    _atomic_write_npz(
        path,
        query_ids=np.asarray(query_ids, dtype=np.str_),
        vectors=vectors,
    )
    return vectors
