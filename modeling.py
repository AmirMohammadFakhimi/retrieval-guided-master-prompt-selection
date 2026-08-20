import copy
import gc
import hashlib
import json
import os
import random
import tempfile
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import lancedb
import numpy as np
import torch
from lancedb.table import LanceTable
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM as AutoCausalLanguageModel,
    AutoModelForImageTextToText as AutoMultimodalLanguageModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from dataset import (
    Column,
    TARGET_TO_AUDIT_COLUMN,
    display_column_name,
)

RETRIEVAL_METHODS = frozenset({'semantic', 'balanced_semantic'})
EXAMPLE_ORDERS = frozenset({
    'most_similar_first',
    'most_similar_last',
    'shuffle',
})
LANCEDB_INGEST_BATCH_SIZE = 2048
EXACT_RETRIEVAL_QUERY_BATCH_SIZE = 64
TRAINING_TABLE_VERSION = 1
QUERY_VECTOR_CACHE_VERSION = 1
TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
LANGUAGE_MODEL_DTYPES = frozenset({*TORCH_DTYPES, 'auto'})
GENERATED_OUTPUT_MAX_NEW_TOKENS = 32
UNBOUNDED_TOKENIZER_LENGTH = 1_000_000_000


class GeneratedOutputError(ValueError):
    """Report which item in a generated-output batch failed validation."""

    def __init__(self, batch_index: int, message: str) -> None:
        super().__init__(message)
        self.batch_index = batch_index


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


def _language_model_context_length(tokenizer: PreTrainedTokenizerBase, language_model: PreTrainedModel) -> int:
    """Return the tokenizer limit or the underlying text-model limit."""

    if tokenizer.model_max_length < UNBOUNDED_TOKENIZER_LENGTH:
        return tokenizer.model_max_length

    return language_model.config.text_config.max_position_embeddings


def choose_device(requested: str = 'auto') -> str:
    """Choose CUDA, then Apple MPS, then CPU."""

    if requested not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError('inference.device must be auto, cuda, mps, or cpu')

    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is not available')

    return requested


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


def clear_model_memory(device: str) -> None:
    """Collect released model objects and clear the active accelerator cache."""

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


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
    query_vectors = load_or_encode_embedding_queries(
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


def render_input(row: dict[str, Any], target: Column) -> str:
    """Render the biography plus the audit column."""

    audit_column = TARGET_TO_AUDIT_COLUMN[target]
    return (
        f'Biography: {row[Column.HARD_TEXT]}\n'
        f'{display_column_name(audit_column)}: {row[audit_column]}'
    )


def build_prompt(
        query: dict[str, Any],
        examples: list[dict[str, Any]],
        target: Column,
        target_labels: list[str],
        master_prompt: str,
) -> list[dict[str, str]]:
    """Build a structured chat with zero or more demonstrations and a target-free query."""

    # Rows contain decoded target-label names, not raw numeric IDs.
    audit_column = TARGET_TO_AUDIT_COLUMN[target]
    target_name = display_column_name(target)
    audit_column_name = display_column_name(audit_column)
    target_labels_text = ', '.join(target_labels)
    try:
        instruction = master_prompt.format(
            target=target_name,
            audit_column=audit_column_name,
            labels=target_labels_text,
        )
    except KeyError as exc:
        raise ValueError('Prompt templates may use only {target}, {audit_column}, and {labels}') from exc

    messages = [{'role': 'system', 'content': instruction}]
    for example in examples:
        messages.extend([
            {'role': 'user', 'content': render_input(example, target)},
            {'role': 'assistant', 'content': example[target]},
        ])

    messages.append({'role': 'user', 'content': render_input(query, target)})

    return messages


def load_language_model(
        language_model_id: str,
        revision: str,
        device: str,
        dtype_name: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load one causal Hugging Face language model."""

    model_config = AutoConfig.from_pretrained(language_model_id, revision=revision)
    model_loader = AutoCausalLanguageModel
    if model_config.model_type == 'mistral3':
        # The official checkpoint wraps its causal text model in a multimodal Mistral3 configuration.
        model_loader = AutoMultimodalLanguageModel

    tokenizer = cast(
        PreTrainedTokenizerBase,
        cast(object, AutoTokenizer.from_pretrained(language_model_id, revision=revision)),
    )

    dtype: str | torch.dtype = 'auto' if dtype_name == 'auto' else TORCH_DTYPES[dtype_name]
    language_model = model_loader.from_pretrained(language_model_id, revision=revision, dtype=dtype)

    language_model.to(device)
    language_model.eval()

    return tokenizer, language_model


def _apply_chat_template(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """Render structured messages directly as one unpadded token sequence."""

    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError(f'{tokenizer.name_or_path} must provide a chat template for this experiment')

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
        add_generation_prompt=False,
        enable_thinking=False,
        preserve_thinking=False,
    )

    if not isinstance(encoded, Mapping):
        raise TypeError('The chat template must return a token dict')

    input_ids = encoded['input_ids']
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f'The chat template returned input_ids with shape {tuple(input_ids.shape)}; expected (1, sequence_length)'
        )

    return input_ids[0]


def _prepare_allowed_label_token_ids(
        messages: list[dict[str, str]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the common assistant prefix and exact content tokens for every label."""

    if not allowed_labels:
        raise ValueError('At least one allowed label is required')

    context_length = _language_model_context_length(tokenizer, language_model)
    if language_model.config.model_type == 'mistral3':
        # Mistral rejects an empty assistant message, but its completed answers end with EOS.
        empty_answer_ids = torch.cat((
            _apply_chat_template(messages, tokenizer),
            torch.tensor([tokenizer.eos_token_id], dtype=torch.long),
        ))
    else:
        empty_answer_ids = _apply_chat_template(
            [
                *messages,
                {'role': 'assistant', 'content': ''}
            ],
            tokenizer,
        )
    empty_token_ids: list[int] = empty_answer_ids.tolist()

    reference_prompt_ids: torch.Tensor | None = None
    candidate_ids_by_label: dict[str, torch.Tensor] = {}
    for label in allowed_labels:
        full_answer_ids = _apply_chat_template(
            [
                *messages,
                {'role': 'assistant', 'content': label}
            ],
            tokenizer,
        )
        full_token_ids: list[int] = full_answer_ids.tolist()

        if len(full_token_ids) <= len(empty_token_ids):
            raise RuntimeError(f'Allowed label {label} did not add tokens to the formatted chat')

        prefix_length = 0
        while prefix_length < len(empty_token_ids) and empty_token_ids[prefix_length] == full_token_ids[prefix_length]:
            prefix_length += 1

        suffix_length = 0
        while (
                suffix_length < len(empty_token_ids) - prefix_length
                and empty_token_ids[-1 - suffix_length] == full_token_ids[-1 - suffix_length]
        ):
            suffix_length += 1

        if prefix_length + suffix_length != len(empty_token_ids):
            raise RuntimeError(
                f'{tokenizer.name_or_path} does not expose an unambiguous assistant-content span through its chat template'
            )

        candidate_stop = len(full_token_ids) - suffix_length
        prompt_ids = full_answer_ids[:prefix_length]
        candidate_ids = full_answer_ids[prefix_length:candidate_stop]

        scored_sequence_length = prefix_length + candidate_ids.numel()
        if scored_sequence_length > context_length:
            raise ValueError(
                f'The formatted prompt plus candidate label {label!r} contains {scored_sequence_length} tokens, '
                f'exceeding {tokenizer.name_or_path}\'s {context_length}-token context limit. '
                f'Shorten the prompt or reduce retrieval.example_counts.'
            )

        if prompt_ids.numel() == 0:
            raise ValueError('The formatted chat produced no prompt tokens')
        if candidate_ids.numel() == 0:
            raise ValueError(f'Allowed label {label} produced no tokens')
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
        elif not torch.equal(reference_prompt_ids, prompt_ids):
            raise RuntimeError('The chat template produced label-dependent prompt tokens')

        candidate_ids_by_label[label] = candidate_ids

    if reference_prompt_ids is None:
        raise RuntimeError('The chat template produced no reference prompt')

    return reference_prompt_ids, candidate_ids_by_label


def generate_allowed_labels(
        message_batch: list[list[dict[str, str]]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
        device: str,
) -> list[tuple[str, str]]:
    """Generate batched assistant responses and accept only exact configured labels."""

    if not allowed_labels:
        raise ValueError('At least one allowed label is required')
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError(f'{tokenizer.name_or_path} must provide a chat template for this experiment')

    model_inputs = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            message_batch,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
            tokenizer_kwargs={
                'padding_side': 'left',
                'return_attention_mask': True,
            },
        ),
    )
    prompt_width = model_inputs['input_ids'].shape[1]
    prompt_lengths = model_inputs['attention_mask'].sum(dim=1).tolist()

    context_length = _language_model_context_length(tokenizer, language_model)
    for batch_index, prompt_length in enumerate(prompt_lengths):
        generated_sequence_length = int(prompt_length) + GENERATED_OUTPUT_MAX_NEW_TOKENS
        if generated_sequence_length > context_length:
            raise GeneratedOutputError(
                batch_index,
                f'The formatted prompt plus {GENERATED_OUTPUT_MAX_NEW_TOKENS} generated tokens contains '
                f'{generated_sequence_length} tokens, exceeding {tokenizer.name_or_path}\'s '
                f'{context_length}-token context limit. Shorten the prompt or reduce retrieval.example_counts.',
            )

    model_inputs = model_inputs.to(device)

    generation_eos_token_id = language_model.generation_config.eos_token_id
    if language_model.config.model_type == 'olmo3':
        # OLMo demonstrations end with <|im_end|>, which its final generation config omits.
        end_of_turn_token_id = tokenizer.get_added_vocab().get('<|im_end|>')
        if end_of_turn_token_id is not None:
            generation_eos_token_id = [generation_eos_token_id, end_of_turn_token_id]
        else:
            raise RuntimeError(f'{tokenizer.name_or_path} does not define the OLMo end-of-turn token <|im_end|>')

    with torch.inference_mode():
        generated_ids = language_model.generate(
            **model_inputs,
            do_sample=False,
            eos_token_id=generation_eos_token_id,
            max_length=None,
            max_new_tokens=GENERATED_OUTPUT_MAX_NEW_TOKENS,
            return_dict_in_generate=False,
        )

    new_token_ids = generated_ids[:, prompt_width:]
    raw_outputs = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    predictions: list[tuple[str, str]] = []
    for batch_index, raw_output in enumerate(raw_outputs):
        predicted_label = raw_output.strip().lower()

        if predicted_label not in allowed_labels:
            raise GeneratedOutputError(
                batch_index,
                f'Language model generated {raw_output!r}; expected exactly one of {allowed_labels!r} '
                f'after trimming surrounding whitespace and converting to lowercase',
            )
        predictions.append((predicted_label, raw_output))

    return predictions


def score_allowed_labels(
        messages: list[dict[str, str]],
        allowed_labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        language_model: PreTrainedModel,
        device: str,
) -> tuple[str, dict[str, float]]:
    """Choose among allowed labels using mean conditional token log-probability.

    Each label is rendered as the final assistant message. Comparing that
    rendering with an empty assistant message isolates the exact
    language-model-specific assistant prefix, label tokens, and end markers.
    Mean rather than summed log-probability reduces the automatic disadvantage
    of labels that use more tokenizer tokens. The common prompt is evaluated
    once, then its inference cache is copied for each multi-token label so only
    that label's remaining tokens require additional model work.
    """

    reference_prompt_ids, candidate_ids_by_label = _prepare_allowed_label_token_ids(
        messages,
        allowed_labels,
        tokenizer,
        language_model,
    )

    text_config = getattr(language_model.config, 'text_config', language_model.config)
    if getattr(text_config, 'num_kv_shared_layers', 0) > 0:
        raise RuntimeError('Inference-cache reuse does not support language models with shared KV layers')

    prompt_input_ids = reference_prompt_ids.unsqueeze(0).to(device)

    with torch.inference_mode():
        prompt_output = language_model(
            input_ids=prompt_input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        if prompt_output.logits.shape[:2] != (1, 1):
            raise RuntimeError(
                f'Could not align language model logits for the common prompt; '
                f'received shape {tuple(prompt_output.logits.shape)}'
            )
        if prompt_output.past_key_values is None:
            raise RuntimeError('The language model did not return an inference cache for the common prompt')

        first_token_log_probabilities = torch.log_softmax(prompt_output.logits[0, 0].float(), dim=-1)

        scores: dict[str, float] = {}
        for label in allowed_labels:
            candidate_ids = candidate_ids_by_label[label]
            candidate_on_device = candidate_ids.to(device)
            selected_token_log_probabilities = first_token_log_probabilities[candidate_on_device[0]].unsqueeze(0)

            if candidate_ids.numel() > 1:
                candidate_cache = copy.deepcopy(prompt_output.past_key_values)
                continuation_ids = candidate_on_device[:-1].unsqueeze(0)
                attention_mask = torch.ones(
                    (1, reference_prompt_ids.numel() + continuation_ids.shape[1]),
                    dtype=torch.long,
                    device=device,
                )
                continuation_output = language_model(
                    input_ids=continuation_ids,
                    attention_mask=attention_mask,
                    past_key_values=candidate_cache,
                    use_cache=False,
                    logits_to_keep=continuation_ids.shape[1],
                )
                continuation_logits = continuation_output.logits[0]

                if continuation_logits.shape[0] != continuation_ids.shape[1]:
                    raise RuntimeError(f'Could not align language model logits for allowed label {label}')

                continuation_log_probabilities = torch.log_softmax(continuation_logits.float(), dim=-1)
                selected_continuation_log_probabilities = continuation_log_probabilities.gather(
                    1, candidate_on_device[1:].unsqueeze(1)
                ).squeeze(1)

                selected_token_log_probabilities = torch.cat((
                    selected_token_log_probabilities,
                    selected_continuation_log_probabilities,
                ))

                del continuation_output, candidate_cache

            score = selected_token_log_probabilities.mean().item()

            if not np.isfinite(score):
                raise RuntimeError(f'Language model returned a non-finite score for allowed label {label}')

            scores[label] = score

    predicted_label = max(allowed_labels, key=lambda label: scores[label])
    return predicted_label, scores
