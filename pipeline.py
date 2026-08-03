import gc
import inspect
import json
import random
import re
import shlex
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast

import lancedb
import numpy as np
import pandas as pd
import torch
import yaml
from bs4 import BeautifulSoup
from datasets import load_dataset
from ftfy import fix_text
from lancedb.table import LanceTable
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

PROFESSIONS = [
    'accountant',
    'architect',
    'attorney',
    'chiropractor',
    'comedian',
    'composer',
    'dentist',
    'dietitian',
    'dj',
    'filmmaker',
    'interior_designer',
    'journalist',
    'model',
    'nurse',
    'painter',
    'paralegal',
    'pastor',
    'personal_trainer',
    'photographer',
    'physician',
    'poet',
    'professor',
    'psychologist',
    'rapper',
    'software_engineer',
    'surgeon',
    'teacher',
    'yoga_teacher',
]
GENDERS = ('male', 'female')
RETRIEVAL_METHODS = frozenset({'semantic', 'balanced_semantic'})
EXAMPLE_ORDERS = frozenset({'as_retrieved', 'reverse', 'shuffle'})
LANCEDB_INGEST_BATCH_SIZE = 2048
TORCH_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}
LANGUAGE_MODEL_DTYPES = frozenset({*TORCH_DTYPES, 'auto'})


class SemanticResource(TypedDict):
    """LanceDB table and matching evaluation embeddings for one model."""

    table: LanceTable
    validation_vectors: np.ndarray
    test_vectors: np.ndarray


class Column(str, Enum):
    """Canonical Bias-in-Bios dataset columns."""

    ID = 'id'
    SPLIT = 'split'
    HARD_TEXT = 'hard_text'
    PROFESSION = 'profession'
    GENDER = 'gender'

    def __str__(self) -> str:
        return self.value


TARGET_TO_OTHER_COLUMN = {
    Column.PROFESSION: Column.GENDER,
    Column.GENDER: Column.PROFESSION,
}

CONDITION_COLUMNS = [
    'condition',
    'evaluation_split',
    'target',
    'audit_column',
    'retrieval',
    'embedding_model',
    'k',
    'example_order',
    'prompt_name',
    'model',
]

INVISIBLE_PATTERN = re.compile(r'[\u00ad\u200b-\u200f\u2060\ufeff]')
CONTROL_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
WHITESPACE_PATTERN = re.compile(r'\s+')
SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r'\s+([,.;:!?])')
HTML_MARKUP_PATTERN = re.compile(
    r'<!--.*?-->|<!doctype\s+[^>]*>|'
    r'</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?/?>',
    re.IGNORECASE | re.DOTALL,
)


def load_config(source: Path) -> dict[str, Any]:
    """Load config.yaml or accept an equivalent dictionary."""

    if isinstance(source, Mapping):
        return dict(source)

    with Path(source).open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    return config


def task_settings(config: dict[str, Any]) -> tuple[Column, Column, list[str], list[str]]:
    """Return target, visible structured column, professions, and target labels."""

    try:
        target = Column(config['defaults']['target'])
        other_column = TARGET_TO_OTHER_COLUMN[target]
    except (KeyError, ValueError) as exc:
        raise ValueError('defaults.target must be profession or gender') from exc

    configured_professions = config['dataset']['professions']

    if configured_professions == 'all':
        professions = list(PROFESSIONS)
    elif isinstance(configured_professions, list):
        professions = configured_professions.copy()
    else:
        raise ValueError('dataset.professions must be a list or the string "all"')

    if not professions:
        raise ValueError('dataset.professions cannot be empty')

    profession_set = set(professions)

    unknown_professions = sorted(profession_set - set(PROFESSIONS))
    if unknown_professions:
        raise ValueError(f'Unknown Bias-in-Bios professions: {unknown_professions}')
    if len(professions) < 2:
        raise ValueError('dataset.professions must contain at least two professions')
    if len(professions) != len(profession_set):
        raise ValueError('dataset.professions cannot contain duplicates')

    target_labels = professions if target is Column.PROFESSION else list(GENDERS)
    return target, other_column, professions, target_labels


def train_size_limit(config: dict[str, Any]) -> int | None:
    """Return the configured training-row cap, or None when all rows are used."""

    configured_train_size = config['dataset']['train_size']
    if configured_train_size == 'all':
        return None
    if (
            isinstance(configured_train_size, bool)
            or not isinstance(configured_train_size, int)
            or configured_train_size < 1
    ):
        raise ValueError('dataset.train_size must be a positive integer or the string "all"')
    return configured_train_size


def validate_config(config: dict[str, Any]) -> None:
    """Fail early for the small set of settings that would invalidate a run."""

    task_settings(config)
    train_size = train_size_limit(config)
    defaults = config['defaults']
    retrieval = config['retrieval']
    model_settings = config['model']

    if defaults['seed'] < 0:
        raise ValueError('defaults.seed must be non-negative')
    if defaults['ranking_direction'] not in {'maximize', 'minimize'}:
        raise ValueError('defaults.ranking_direction must be maximize or minimize')
    if config['dataset']['validation_per_profession_gender'] < 1:
        raise ValueError('dataset.validation_per_profession_gender must be at least 1')
    if config['dataset']['test_per_profession_gender'] < 1:
        raise ValueError('dataset.test_per_profession_gender must be at least 1')
    if config['dataset']['shuffle_seed'] < 0:
        raise ValueError('dataset.shuffle_seed must be non-negative')
    language_models = model_settings['language_models']
    if not isinstance(language_models, list) or not language_models:
        raise ValueError('model.language_models must contain at least one model')

    language_model_ids: list[str] = []
    for index, language_model in enumerate(language_models):
        if not isinstance(language_model, dict):
            raise ValueError(f'model.language_models[{index}] must be a mapping')

        missing_settings = sorted({'id', 'revision', 'dtype'} - set(language_model))
        if missing_settings:
            raise ValueError(
                f'model.language_models[{index}] is missing: '
                f'{missing_settings}'
            )

        model_id = language_model['id']
        if not model_id:
            raise ValueError(
                f'model.language_models[{index}].id cannot be empty'
            )
        language_model_ids.append(model_id)
        if not language_model['revision']:
            raise ValueError(f'model.language_models[{index}].revision cannot be empty')
        if language_model['dtype'] not in LANGUAGE_MODEL_DTYPES:
            allowed_dtypes = ', '.join(sorted(LANGUAGE_MODEL_DTYPES))
            raise ValueError(
                f'model.language_models[{index}].dtype must be one of: {allowed_dtypes}'
            )
    if len(language_model_ids) != len(set(language_model_ids)):
        raise ValueError('model.language_models cannot contain duplicate IDs')

    methods = retrieval['methods']
    if not isinstance(methods, list) or not methods:
        raise ValueError('retrieval.methods must be a non-empty list')
    unknown_methods = sorted(set(methods) - RETRIEVAL_METHODS)
    if unknown_methods:
        raise ValueError(
            f'Unknown retrieval methods {unknown_methods}; expected values from {sorted(RETRIEVAL_METHODS)}'
        )
    if len(methods) != len(set(methods)):
        raise ValueError('retrieval.methods cannot contain duplicates')
    k_values = [value for value in retrieval['k_values']]
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError('retrieval.k_values must contain positive integers')
    if train_size is not None and max(k_values) > train_size:
        raise ValueError('Every retrieval.k_values entry must be <= dataset.train_size')
    if len(k_values) != len(set(k_values)):
        raise ValueError('retrieval.k_values cannot contain duplicates')
    if not str(retrieval['lancedb_path']):
        raise ValueError('retrieval.lancedb_path cannot be empty')

    embedding_models = retrieval['embedding_models']
    if not isinstance(embedding_models, list) or not embedding_models:
        raise ValueError('retrieval.embedding_models must contain at least one model')
    required_embedding_settings = {
        'id',
        'dimension',
        'max_sequence_length',
        'batch_size',
        'dtype',
        'query_prompt',
    }
    embedding_model_ids: list[str] = []
    for index, embedding_model in enumerate(embedding_models):
        if not isinstance(embedding_model, dict):
            raise ValueError(f'retrieval.embedding_models[{index}] must be a mapping')
        missing_settings = sorted(required_embedding_settings - set(embedding_model))
        if missing_settings:
            raise ValueError(f'retrieval.embedding_models[{index}] is missing: {missing_settings}')
        model_id = embedding_model['id']
        if not model_id:
            raise ValueError(f'retrieval.embedding_models[{index}].id cannot be empty')
        embedding_model_ids.append(model_id)
        for setting in ('dimension', 'max_sequence_length', 'batch_size'):
            value = embedding_model[setting]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'retrieval.embedding_models[{index}].{setting} must be a positive integer')
        if embedding_model['dtype'] not in TORCH_DTYPES:
            allowed_dtypes = ', '.join(TORCH_DTYPES)
            raise ValueError(f'retrieval.embedding_models[{index}].dtype must be one of: {allowed_dtypes}')
        if not embedding_model['query_prompt']:
            raise ValueError(f'retrieval.embedding_models[{index}].query_prompt cannot be empty')
    if len(embedding_model_ids) != len(set(embedding_model_ids)):
        raise ValueError('retrieval.embedding_models cannot contain duplicate IDs')

    orders = retrieval['example_orders']
    if not isinstance(orders, list) or not orders:
        raise ValueError('retrieval.example_orders must be a non-empty list')
    unknown_orders = sorted(set(orders) - EXAMPLE_ORDERS)
    if unknown_orders:
        raise ValueError(f'Unknown example orders {unknown_orders}; expected values from {sorted(EXAMPLE_ORDERS)}')
    if len(orders) != len(set(orders)):
        raise ValueError('retrieval.example_orders cannot contain duplicates')

    templates = config['prompt_templates']
    if not isinstance(templates, dict) or not templates:
        raise ValueError('prompt_templates must contain at least one named prompt')
    if not all(name and text for name, text in templates.items()):
        raise ValueError('prompt template names and texts cannot be empty')


def choose_device(requested: str = 'auto') -> str:
    """Choose CUDA, then Apple MPS, then CPU."""

    if requested not in {'auto', 'cuda', 'mps', 'cpu'}:
        raise ValueError('model.device must be auto, cuda, mps, or cpu')

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


def _clean_text(value: str) -> str:
    """Repair technical corruption without removing linguistic signals."""

    text = value
    if HTML_MARKUP_PATTERN.search(value):
        soup = BeautifulSoup(value, 'html.parser')
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        text = soup.get_text(separator=' ')

    text = fix_text(
        text,
        unescape_html=True,
        normalization='NFC',
    )

    text = INVISIBLE_PATTERN.sub('', text)
    text = CONTROL_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    text = SPACE_BEFORE_PUNCTUATION_PATTERN.sub(r'\1', text)

    return text


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                Column.ID.value: row[Column.ID],
                Column.SPLIT.value: row[Column.SPLIT],
                Column.HARD_TEXT.value: row[Column.HARD_TEXT],
                Column.PROFESSION.value: row[Column.PROFESSION],
                Column.GENDER.value: row[Column.GENDER],
            })

    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _download_data(config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    """Download, shuffle, clean, and cache every official source split."""

    dataset_config = config['dataset']
    shuffle_seed = int(dataset_config['shuffle_seed'])
    rows: list[dict[str, Any]] = []

    for split_name in ('train', 'dev', 'test'):
        split_data = load_dataset(dataset_config['hub_id'], split=split_name).shuffle(seed=shuffle_seed)

        if len(split_data) == 0:
            raise ValueError(f'Bias in Bios {split_name} split is empty')
        for index, row in enumerate(split_data):
            rows.append({
                Column.ID.value: f'{split_name}:{shuffle_seed}:{index}',
                Column.SPLIT.value: split_name,
                Column.HARD_TEXT.value: _clean_text(row[Column.HARD_TEXT]),
                Column.PROFESSION.value: PROFESSIONS[row[Column.PROFESSION]],
                Column.GENDER.value: GENDERS[row[Column.GENDER]],
            })

    _write_jsonl(destination, rows)
    return rows


def load_data(config: dict[str, Any], project_root: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    """Load retrieval-train, prompt-selection, and final-test rows."""

    dataset_config = config['dataset']
    dataset_path = project_root / dataset_config['file']
    train_size = train_size_limit(config)
    if not dataset_path.exists():
        rows = _download_data(config, dataset_path)
    else:
        rows = _read_jsonl(dataset_path)

    _, _, professions, _ = task_settings(config)
    rows = [row for row in rows if row[Column.PROFESSION] in professions]

    available_train = [row for row in rows if row[Column.SPLIT] == 'train']
    train = available_train if train_size is None else available_train[:train_size]

    if train_size is not None and len(train) < train_size:
        raise ValueError(
            f'Only {len(available_train)} matching training rows exist; '
            f'dataset.train_size requested {train_size}'
        )
    if not train:
        raise ValueError('The training demonstration pool is empty')
    if int(max(config['retrieval']['k_values'])) > len(train):
        raise ValueError(
            'Every retrieval.k_values entry must be <= the available '
            f'training pool size ({len(train)})'
        )

    validation_per_cell = dataset_config['validation_per_profession_gender']
    test_per_cell = dataset_config['test_per_profession_gender']
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    missing_cells: list[tuple[str, str, str]] = []

    progress_bar = tqdm(total=len(professions) * len(GENDERS), desc='Selecting evaluation cells', unit='cell')
    for profession in professions:
        for gender in GENDERS:
            validation_cell = [
                row
                for row in rows
                if row[Column.SPLIT] == 'dev'
                   and row[Column.PROFESSION] == profession
                   and row[Column.GENDER] == gender
            ][:validation_per_cell]

            test_cell = [
                row
                for row in rows
                if row[Column.SPLIT] == 'test'
                   and row[Column.PROFESSION] == profession
                   and row[Column.GENDER] == gender
            ][:test_per_cell]

            validation.extend(
                {**row, Column.SPLIT.value: 'validation'}
                for row in validation_cell
            )
            test.extend(
                {**row, Column.SPLIT.value: 'test'}
                for row in test_cell
            )

            if len(validation_cell) < validation_per_cell:
                missing_cells.append(('dev', profession, gender))
            if len(test_cell) < test_per_cell:
                missing_cells.append(('test', profession, gender))
            progress_bar.update(1)

    if missing_cells:
        raise ValueError(f'Bias in Bios lacks enough rows for these split/profession/gender cells: {missing_cells}')

    counts = (
        pd.DataFrame(train + validation + test)
        .groupby([Column.SPLIT, Column.PROFESSION, Column.GENDER], as_index=False)
        .size()
        .rename(columns={'size': 'count'})
    )
    complete_cells = pd.MultiIndex.from_product(
        [['train', 'validation', 'test'], professions, GENDERS],
        names=[Column.SPLIT, Column.PROFESSION, Column.GENDER],
    )
    counts = (
        counts.set_index([Column.SPLIT, Column.PROFESSION, Column.GENDER])
        .reindex(complete_cells, fill_value=0)
        .reset_index()
    )

    counts['gender_share_within_profession'] = counts['count'] / counts.groupby(
        [Column.SPLIT, Column.PROFESSION]
    )['count'].transform('sum')
    counts['profession_share_within_gender'] = counts['count'] / counts.groupby(
        [Column.SPLIT, Column.GENDER]
    )['count'].transform('sum')
    counts['cell_share_of_split'] = counts['count'] / counts.groupby(
        Column.SPLIT
    )['count'].transform('sum')
    counts['gender_share_gap_within_profession'] = counts.groupby(
        [Column.SPLIT, Column.PROFESSION]
    )['gender_share_within_profession'].transform(lambda values: values.max() - values.min())
    counts['profession_share_gap_within_gender'] = counts.groupby(
        [Column.SPLIT, Column.GENDER]
    )['profession_share_within_gender'].transform(lambda values: values.max() - values.min())

    return train, validation, test, counts


def prepare_semantic_retrieval(
        train: list[dict[str, Any]],
        queries: list[dict[str, Any]],
        model_settings: dict[str, Any],
        device: str,
        database_path: Path,
        progress: Callable[[str], None] = print,
) -> tuple[LanceTable, np.ndarray]:
    """Open or build the persistent LanceDB training table and encode queries."""

    model_name = model_settings['id']
    embedding_dimension = model_settings['dimension']
    max_sequence_length = model_settings['max_sequence_length']
    batch_size = model_settings['batch_size']
    dtype_name = model_settings['dtype']
    query_prompt = model_settings['query_prompt']
    database = lancedb.connect(database_path)
    table_name = f'semantic_{model_name.lower().replace("/", "_")}'

    if table_name in database.list_tables().tables:
        table = database.open_table(table_name)
        if table.count_rows() != len(train):
            raise RuntimeError(
                f'LanceDB table {table_name} does not match the configured '
                'training pool. Delete the LanceDB directory and rerun from '
                'the beginning:\n'
                f'rm -rf -- {shlex.quote(str(database_path))}'
            )
        progress(
            f'Reusing {len(train)} cached training embeddings from '
            f'{database_path / table_name}'
        )
    else:
        table = None

    encoder = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={'dtype': TORCH_DTYPES[dtype_name]},
        truncate_dim=embedding_dimension,
    )
    encoder.max_seq_length = max_sequence_length

    if table is None:
        progress(
            f'Building LanceDB table {table_name} with '
            f'{len(train)} training embeddings'
        )
        progress_bar = tqdm(
            total=len(train),
            desc='Embedding training rows for LanceDB',
            unit='row',
        )

        try:
            for start in range(0, len(train), LANCEDB_INGEST_BATCH_SIZE):
                batch = train[start:start + LANCEDB_INGEST_BATCH_SIZE]
                vectors = np.asarray(
                    encoder.encode(
                        [row[Column.HARD_TEXT] for row in batch],
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                )

                if vectors.ndim != 2 or vectors.shape[1] != embedding_dimension:
                    raise RuntimeError(
                        f'Embedding model returned vectors with shape '
                        f'{vectors.shape}; expected (*, {embedding_dimension})'
                    )

                records = [
                    {
                        Column.ID.value: row[Column.ID],
                        Column.SPLIT.value: row[Column.SPLIT],
                        Column.HARD_TEXT.value: row[Column.HARD_TEXT],
                        Column.PROFESSION.value: row[Column.PROFESSION],
                        Column.GENDER.value: row[Column.GENDER],
                        'vector': vector.tolist(),
                    }
                    for row, vector in zip(batch, vectors, strict=True)
                ]

                if table is None:
                    table = database.create_table(table_name, data=records)
                else:
                    cast(LanceTable, table).add(records)

                progress_bar.update(len(batch))
            progress_bar.close()

        except Exception:
            progress_bar.close()
            if table_name in database.list_tables().tables:
                database.drop_table(table_name)
            raise

    if table is None:
        raise RuntimeError('Cannot build a LanceDB table from an empty training pool')

    query_vectors = np.asarray(
        encoder.encode(
            [row[Column.HARD_TEXT] for row in queries],
            prompt=query_prompt,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    )

    if query_vectors.ndim != 2 or query_vectors.shape != (len(queries), embedding_dimension):
        raise RuntimeError(
            f'Embedding model returned query vectors with shape '
            f'{query_vectors.shape}; expected '
            f'({len(queries)}, {embedding_dimension})'
        )

    del encoder
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return cast(LanceTable, table), query_vectors


def _get_semantic_candidate_page(
        semantic_table: LanceTable,
        query_vector: np.ndarray,
        limit: int,
        offset: int = 0,
) -> list[dict[str, Any]]:
    """Return exact cosine-nearest LanceDB rows in semantic rank order."""

    rows = (
        semantic_table.search(query_vector)
        .distance_type('cosine')
        .bypass_vector_index()
        .select([
            Column.ID.value,
            Column.SPLIT.value,
            Column.HARD_TEXT.value,
            Column.PROFESSION.value,
            Column.GENDER.value,
            '_distance',
        ])
        .limit(limit)
        .offset(offset)
        .to_list()
    )

    candidates: list[dict[str, Any]] = []
    for row in rows:
        distance = float(row.pop('_distance'))
        candidates.append({**row, 'retrieval_score': 1.0 - distance})

    return candidates


def _get_balanced_semantic_candidates(
        semantic_table: LanceTable,
        query_vector: np.ndarray,
        k: int,
        cells: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Select the nearest candidates while keeping all group counts balanced."""

    if not cells:
        raise ValueError('Balanced semantic retrieval requires training cells')

    professions = tuple(sorted({profession for profession, _ in cells}))
    genders = tuple(sorted({gender for _, gender in cells}))

    profession_counts = Counter()
    gender_counts = Counter()
    cell_counts = Counter()

    unselected_candidates: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []

    row_count = semantic_table.count_rows()
    offset = 0
    page_size = max(32, 8 * k)

    while len(selected) < k:
        minimum_profession_count = min(profession_counts[profession] for profession in professions)
        minimum_gender_count = min(gender_counts[gender] for gender in genders)
        minimum_cell_count = min(cell_counts[cell] for cell in cells)

        first_eligible_index = next(
            (
                index
                for index, candidate in enumerate(unselected_candidates)
                if profession_counts[candidate[Column.PROFESSION]] == minimum_profession_count
                   and gender_counts[candidate[Column.GENDER]] == minimum_gender_count
                   and cell_counts[(candidate[Column.PROFESSION], candidate[Column.GENDER])] == minimum_cell_count
            ),
            None,
        )

        if first_eligible_index is None:
            if offset >= row_count:
                break

            page = _get_semantic_candidate_page(
                semantic_table,
                query_vector,
                limit=min(page_size, row_count - offset),
                offset=offset,
            )
            if not page:
                break

            unselected_candidates.extend(page)
            offset += len(page)
            page_size *= 2
            continue

        candidate = unselected_candidates.pop(first_eligible_index)
        profession = candidate[Column.PROFESSION]
        gender = candidate[Column.GENDER]

        selected.append(candidate)
        profession_counts[profession] += 1
        gender_counts[gender] += 1
        cell_counts[(profession, gender)] += 1

    if len(selected) != k:
        raise RuntimeError(
            f'Could not select {k} balanced semantic examples after scanning {offset} of {row_count} training rows'
        )

    return selected


def retrieve_examples(
        method: str,
        query_vector: np.ndarray,
        semantic_table: LanceTable,
        k: int,
        cells: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Retrieve exact semantic or relevance-first balanced examples."""

    if method not in RETRIEVAL_METHODS:
        raise ValueError(f'Unknown retrieval method {method!r}; expected one of {sorted(RETRIEVAL_METHODS)}')

    row_count = semantic_table.count_rows()
    if k > row_count:
        raise ValueError(f'Cannot retrieve {k} examples from {row_count} training rows')

    if method == 'semantic':
        candidates = _get_semantic_candidate_page(semantic_table, query_vector, k)
        if len(candidates) != k:
            raise RuntimeError(f'LanceDB returned {len(candidates)} candidates; expected {k}')
        return candidates

    return _get_balanced_semantic_candidates(semantic_table, query_vector, k, cells)


def order_examples(
        examples: list[dict[str, Any]],
        order: str,
        seed: int,
) -> list[dict[str, Any]]:
    """Apply the configured demonstration order after retrieval."""

    if order not in EXAMPLE_ORDERS:
        raise ValueError(f'Unknown example order {order!r}; expected one of {sorted(EXAMPLE_ORDERS)}')

    if order == 'as_retrieved':
        return examples

    if order == 'reverse':
        return examples[::-1]

    if order == 'shuffle':
        random.Random(seed).shuffle(examples)
        return examples

    raise RuntimeError(f'Example order {order!r} is allowed but not implemented')


def display_column_name(column: Column) -> str:
    """Return a human-readable name for a dataset column used in prompts."""

    return str(column).replace('_', ' ').title()


def render_input(row: dict[str, Any], target: Column) -> str:
    """Render the biography plus the non-target structured column."""

    other_column = TARGET_TO_OTHER_COLUMN[target]
    return (
        f'Biography: {row[Column.HARD_TEXT]}\n'
        f'{display_column_name(other_column)}: {row[other_column]}'
    )


def build_prompt(
        query: dict[str, Any],
        examples: list[dict[str, Any]],
        target: Column,
        labels: list[str],
        master_prompt: str,
) -> list[dict[str, str]]:
    """Build a structured chat with demonstrations and a target-free query."""

    # Rows must contain decoded class names from load_data, not raw numeric IDs.
    other_column = TARGET_TO_OTHER_COLUMN[target]
    target_name = display_column_name(target)
    other_column_name = display_column_name(other_column)
    labels_text = ', '.join(labels)
    try:
        instruction = master_prompt.format(
            target=target_name,
            other_column=other_column_name,
            labels=labels_text,
        )
    except KeyError as exc:
        raise ValueError(f'Prompt templates may use only {target}, {other_column}, and {labels}') from exc

    instruction_block = (
        f'{instruction}\n'
        f'Allowed values for {target_name}: {labels_text}.\n'
        'Output exactly one allowed value and no explanation.'
    )

    messages = [
        {'role': 'system', 'content': instruction_block}
    ]
    for example in examples:
        messages.extend([
            {'role': 'user', 'content': render_input(example, target)},
            {'role': 'assistant', 'content': example[target]},
        ])
    messages.append(
        {'role': 'user', 'content': render_input(query, target)}
    )
    return messages


def load_llm(
        model_id: str,
        revision: str,
        device: str,
        dtype_name: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load one causal or multimodal Hugging Face language model."""

    tokenizer = cast(
        PreTrainedTokenizerBase,
        cast(object, AutoTokenizer.from_pretrained(model_id, revision=revision)),
    )

    dtype: str | torch.dtype = 'auto' if dtype_name == 'auto' else TORCH_DTYPES[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)

    model.to(device)
    model.eval()

    return tokenizer, model


def clear_model_memory(device: str) -> None:
    """Collect released model objects and clear the active accelerator cache."""

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


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
    )

    if not isinstance(encoded, dict):
        raise TypeError('The chat template must return a token mapping')

    input_ids = encoded['input_ids']
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise RuntimeError(
            f'The chat template returned input_ids with shape {tuple(input_ids.shape)}; expected (1, sequence_length)'
        )

    return input_ids[0]


def score_allowed_labels(
        messages: list[dict[str, str]],
        labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        device: str,
) -> tuple[str, dict[str, float]]:
    """Choose among allowed labels using mean conditional token log-probability.

    Each label is rendered as the final assistant message. Comparing that
    rendering with an empty assistant message isolates the exact model-specific
    assistant prefix, label tokens, and end markers. Mean rather than summed
    log-probability reduces the automatic disadvantage of labels that use more
    tokenizer tokens. Labels are evaluated sequentially to keep accelerator
    memory small.
    """

    if not labels:
        raise ValueError('At least one allowed label is required')

    empty_answer_ids = _apply_chat_template(
        [
            *messages,
            {'role': 'assistant', 'content': ''}
        ],
        tokenizer,
    )
    empty_token_ids: list[int] = empty_answer_ids.tolist()

    scores: dict[str, float] = {}
    reference_prompt_ids: torch.Tensor | None = None
    for label in labels:
        full_answer_ids = _apply_chat_template(
            [*messages, {'role': 'assistant', 'content': label}],
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

        if prompt_ids.numel() == 0:
            raise ValueError('The formatted chat produced no prompt tokens')
        if candidate_ids.numel() == 0:
            raise ValueError(f'Allowed label {label} produced no tokens')
        if reference_prompt_ids is None:
            reference_prompt_ids = prompt_ids
        elif not torch.equal(reference_prompt_ids, prompt_ids):
            raise RuntimeError('The chat template produced label-dependent prompt tokens')

        scoring_ids = full_answer_ids[:candidate_stop - 1].unsqueeze(0).to(device)

        with torch.inference_mode():
            # Omitting the final candidate token leaves exactly the C logits
            # that predict the C candidate tokens.
            output = model(input_ids=scoring_ids, use_cache=False, logits_to_keep=candidate_ids.numel())
            candidate_logits = output.logits[0]

            if candidate_logits.shape[0] != candidate_ids.numel():
                raise RuntimeError(f'Could not align model logits for allowed label {label}')

            token_log_probabilities = torch.log_softmax(candidate_logits.float(), dim=-1)
            candidate_on_device = candidate_ids.to(device)
            selected_token_log_probabilities = token_log_probabilities.gather(
                1, candidate_on_device.unsqueeze(1)
            ).squeeze(1)
            score = selected_token_log_probabilities.mean().item()

        if not np.isfinite(score):
            raise RuntimeError(f'Model returned a non-finite score for allowed label {label}')
        scores[label] = score

    predicted_label = max(labels, key=lambda label: scores[label])
    return predicted_label, scores


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else np.nan


def _rate_range(values: list[float]) -> float:
    defined = [value for value in values if not pd.isna(value)]
    return max(defined) - min(defined) if len(defined) >= 2 else np.nan


def calculate_metrics(predictions: pd.DataFrame, labels: list[str]
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate hard-label classification and group-disparity metrics."""

    result_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for _, condition_predictions in predictions.groupby(['evaluation_split', 'condition'], sort=False):
        condition_metadata = {column: condition_predictions[column].iloc[0] for column in CONDITION_COLUMNS}
        true_labels = condition_predictions['true_label']
        predicted_labels = condition_predictions['predicted_label']

        condition_class_rows: list[dict[str, Any]] = []
        for label in labels:
            true_positive = ((true_labels == label) & (predicted_labels == label)).sum()
            false_positive = ((true_labels != label) & (predicted_labels == label)).sum()
            false_negative = ((true_labels == label) & (predicted_labels != label)).sum()
            true_negative = ((true_labels != label) & (predicted_labels != label)).sum()
            support = true_positive + false_negative
            predicted_count = true_positive + false_positive
            precision = true_positive / predicted_count if predicted_count else 0.0
            recall = true_positive / support
            f1 = 2 * true_positive / (
                    2 * true_positive + false_positive + false_negative
            )
            specificity = true_negative / (true_negative + false_positive)
            class_metric_row = {
                **condition_metadata,
                'target_class': label,
                'support': support,
                'predicted_count': predicted_count,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'false_positive_rate': false_positive / (false_positive + true_negative),
                'false_negative_rate': false_negative / (false_negative + true_positive),
                'negative_predictive_value': _safe_rate(true_negative, true_negative + false_negative),
            }
            condition_class_rows.append(class_metric_row)
            class_rows.append(class_metric_row)

        class_frame = pd.DataFrame(condition_class_rows)
        sample_count = len(condition_predictions)
        correct_count = (true_labels == predicted_labels).sum()
        accuracy = correct_count / sample_count
        class_weights = class_frame['support'] / sample_count

        for true_label in labels:
            for predicted_label in labels:
                confusion_rows.append(
                    {
                        **condition_metadata,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'count': ((true_labels == true_label) & (predicted_labels == predicted_label)).sum(),
                    }
                )

        audit_groups = condition_predictions['audit_group'].unique().tolist()
        group_accuracy_values: list[float] = []
        condition_fairness: list[dict[str, Any]] = []

        for target_class in labels:
            class_group_rows: list[dict[str, Any]] = []
            for audit_group in audit_groups:
                group_predictions = condition_predictions[condition_predictions['audit_group'] == audit_group]
                group_truth = group_predictions['true_label']
                group_predicted = predicted_labels.loc[group_predictions.index]
                actual_positive = group_truth == target_class
                predicted_positive = group_predicted == target_class
                tp = (actual_positive & predicted_positive).sum()
                fp = (~actual_positive & predicted_positive).sum()
                fn = (actual_positive & ~predicted_positive).sum()
                tn = (~actual_positive & ~predicted_positive).sum()
                group_n = len(group_predictions)
                group_accuracy = (group_truth == group_predicted).mean()
                group_metric_row = {
                    **condition_metadata,
                    'target_class': target_class,
                    'audit_group': audit_group,
                    'group_n': group_n,
                    'group_accuracy': group_accuracy,
                    'positive_support': tp + fn,
                    'negative_support': fp + tn,
                    'predicted_positive': tp + fp,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'selection_rate': (tp + fp) / group_n,
                    'true_positive_rate': _safe_rate(tp, tp + fn),
                    'false_positive_rate': _safe_rate(fp, fp + tn),
                    'false_negative_rate': _safe_rate(fn, tp + fn),
                    'positive_predictive_value': _safe_rate(tp, tp + fp),
                    'specificity': _safe_rate(tn, tn + fp),
                    'negative_predictive_value': _safe_rate(tn, tn + fn),
                }
                class_group_rows.append(group_metric_row)
                group_rows.append(group_metric_row)

            group_metrics = pd.DataFrame(class_group_rows)
            selection_rates = group_metrics['selection_rate']
            demographic_parity_ratio = selection_rates.min() / selection_rates.max() if selection_rates.max() > 0 else np.nan
            equal_opportunity_difference = _rate_range(group_metrics['true_positive_rate'].tolist())
            false_positive_rate_difference = _rate_range(group_metrics['false_positive_rate'].tolist())
            equalized_odds_difference = np.maximum(equal_opportunity_difference, false_positive_rate_difference)

            fairness_row = {
                **condition_metadata,
                'target_class': target_class,
                'groups_compared': len(audit_groups),
                'selection_rate_groups_defined': (group_metrics['selection_rate'].notna().sum()),
                'tpr_groups_defined': (group_metrics['true_positive_rate'].notna().sum()),
                'fpr_groups_defined': (group_metrics['false_positive_rate'].notna().sum()),
                'ppv_groups_defined': (group_metrics['positive_predictive_value'].notna().sum()),
                'demographic_parity_difference': _rate_range(group_metrics['selection_rate'].tolist()),
                'demographic_parity_ratio': demographic_parity_ratio,
                'equal_opportunity_difference': equal_opportunity_difference,
                'false_positive_rate_difference': false_positive_rate_difference,
                'equalized_odds_difference': equalized_odds_difference,
                'predictive_parity_difference': _rate_range(group_metrics['positive_predictive_value'].tolist()),
            }
            fairness_rows.append(fairness_row)
            condition_fairness.append(fairness_row)

        # Group-wide accuracy does not depend on a target class.
        for audit_group in audit_groups:
            group_predictions = condition_predictions[condition_predictions['audit_group'] == audit_group]
            group_truth = group_predictions['true_label']
            group_predicted = predicted_labels.loc[group_predictions.index]
            group_accuracy_values.append((group_truth == group_predicted).mean())

        condition_fairness_metrics = pd.DataFrame(condition_fairness)
        result_rows.append(
            {
                **condition_metadata,
                'sample_count': sample_count,
                'n_classes': len(labels),
                'n_audit_groups': len(audit_groups),
                'accuracy': accuracy,
                'balanced_accuracy': class_frame['recall'].mean(),
                'macro_precision': class_frame['precision'].mean(),
                'macro_recall': class_frame['recall'].mean(),
                'macro_f1': class_frame['f1'].mean(),
                'micro_precision': accuracy,
                'micro_recall': accuracy,
                'micro_f1': accuracy,
                'weighted_precision': (
                        class_frame['precision'] * class_weights
                ).sum(),
                'weighted_recall': (class_frame['recall'] * class_weights).sum(),
                'weighted_f1': (class_frame['f1'] * class_weights).sum(),
                'matthews_correlation_coefficient': matthews_corrcoef(
                    true_labels, predicted_labels
                ),
                'cohen_kappa': cohen_kappa_score(true_labels, predicted_labels),
                'worst_group_accuracy': min(group_accuracy_values),
                'group_accuracy_difference': _rate_range(group_accuracy_values),
                'mean_demographic_parity_difference': condition_fairness_metrics[
                    'demographic_parity_difference'
                ].mean(),
                'max_demographic_parity_difference': condition_fairness_metrics[
                    'demographic_parity_difference'
                ].max(),
                'mean_demographic_parity_ratio': condition_fairness_metrics[
                    'demographic_parity_ratio'
                ].mean(),
                'min_demographic_parity_ratio': condition_fairness_metrics[
                    'demographic_parity_ratio'
                ].min(),
                'mean_equal_opportunity_difference': condition_fairness_metrics[
                    'equal_opportunity_difference'
                ].mean(),
                'max_equal_opportunity_difference': condition_fairness_metrics[
                    'equal_opportunity_difference'
                ].max(),
                'mean_false_positive_rate_difference': condition_fairness_metrics[
                    'false_positive_rate_difference'
                ].mean(),
                'max_false_positive_rate_difference': condition_fairness_metrics[
                    'false_positive_rate_difference'
                ].max(),
                'mean_equalized_odds_difference': condition_fairness_metrics[
                    'equalized_odds_difference'
                ].mean(),
                'max_equalized_odds_difference': condition_fairness_metrics[
                    'equalized_odds_difference'
                ].max(),
                'n_demographic_parity_defined_classes': condition_fairness_metrics[
                    'demographic_parity_difference'
                ].notna().sum(),
                'n_demographic_parity_ratio_defined_classes': condition_fairness_metrics[
                    'demographic_parity_ratio'
                ].notna().sum(),
                'n_equal_opportunity_defined_classes': condition_fairness_metrics[
                    'equal_opportunity_difference'
                ].notna().sum(),
                'n_false_positive_rate_defined_classes': condition_fairness_metrics[
                    'false_positive_rate_difference'
                ].notna().sum(),
                'n_equalized_odds_defined_classes': condition_fairness_metrics[
                    'equalized_odds_difference'
                ].notna().sum(),
                'n_predictive_parity_defined_classes': condition_fairness_metrics[
                    'predictive_parity_difference'
                ].notna().sum(),
                'mean_predictive_parity_difference': condition_fairness_metrics[
                    'predictive_parity_difference'
                ].mean(),
                'max_predictive_parity_difference': condition_fairness_metrics[
                    'predictive_parity_difference'
                ].max(),
            }
        )

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(fairness_rows),
    )


def rank_results(
        results: pd.DataFrame, metric: str, direction: str
) -> pd.DataFrame:
    """Rank prompt configurations separately within each language model."""

    if metric not in results.columns:
        available = ', '.join(results.select_dtypes(include='number').columns)
        raise ValueError(
            f'Unknown ranking metric {metric!r}. Numeric result columns: {available}'
        )
    if not pd.api.types.is_numeric_dtype(results[metric]):
        raise ValueError(f'Ranking metric {metric!r} must be numeric')
    if results[metric].notna().sum() == 0:
        raise ValueError(f'Ranking metric {metric!r} is undefined for every condition')
    ranked = results.sort_values(
        ['model', metric, 'condition'],
        ascending=[True, direction == 'minimize', True],
        na_position='last',
        kind='stable',
    ).reset_index(drop=True)
    ranked.insert(
        0,
        'rank',
        ranked.groupby('model', sort=False).cumcount() + 1,
    )
    ranked.insert(1, 'is_best', ranked['rank'].eq(1))
    return ranked


def _plot_results(
        validation_results: pd.DataFrame,
        output: Path,
) -> None:
    """Plot validation comparisons; final-test rows are saved separately."""

    import matplotlib.pyplot as plt

    plot_frame = validation_results.sort_values(
        'rank', ascending=False
    ).reset_index(drop=True)
    labels = [
        (
            f'model={row.model}, {row.retrieval}, '
            f'embedding={row.embedding_model}, '
            f'k={row.k}, {row.prompt_name}, {row.example_order}'
        )
        for row in plot_frame.itertuples(index=False)
    ]
    y = np.arange(len(plot_frame))
    height = max(5.0, 0.46 * len(plot_frame) + 1.8)
    figure, axes = plt.subplots(1, 2, figsize=(16, height), sharey=True)

    axes[0].barh(y - 0.18, plot_frame['accuracy'], 0.36, label='Accuracy')
    axes[0].barh(y + 0.18, plot_frame['macro_f1'], 0.36, label='Macro-F1')
    axes[0].set_xlim(0, 1)
    axes[0].set_title('Prediction quality (higher is better)')
    axes[0].legend()

    axes[1].barh(
        y - 0.24,
        plot_frame['max_demographic_parity_difference'],
        0.24,
        label='Demographic parity',
    )
    axes[1].barh(
        y,
        plot_frame['max_equal_opportunity_difference'],
        0.24,
        label='Equal opportunity',
    )
    axes[1].barh(
        y + 0.24,
        plot_frame['max_equalized_odds_difference'],
        0.24,
        label='Equalized odds',
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Maximum group differences (lower is better)')
    axes[1].legend()

    axes[0].set_yticks(y, labels)
    for axis in axes:
        axis.grid(axis='x', alpha=0.2)
        axis.set_xlabel('Score')
    figure.suptitle(
        f'Validation prompt comparison — target: {plot_frame['target'].iloc[0]}, '
        f'audit groups: {plot_frame['audit_column'].iloc[0]}\n'
        f'One validation winner and one final-test row per model'
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _write_best_prompts(
        path: Path,
        selected: pd.DataFrame,
        prompt_templates: Mapping[str, Any],
        labels: list[str],
        ranking_metric: str,
        ranking_direction: str,
        final_results: pd.DataFrame,
) -> None:
    """Save one validation-selected prompt and final-test score per model."""

    sections: list[str] = []
    final_by_model = final_results.set_index('model')
    for best in selected.itertuples(index=False):
        final_result = final_by_model.loc[best.model]
        resolved_prompt = prompt_templates[best.prompt_name].format(
            target=display_column_name(best.target),
            other_column=display_column_name(best.audit_column),
            labels=', '.join(labels),
        )
        sections.append(
            f'Model: {best.model}\n'
            f'Selected on validation metric: {ranking_metric} '
            f'({ranking_direction})\n'
            f'Validation score: {getattr(best, ranking_metric)}\n'
            f'Final test score: {final_result[ranking_metric]}\n'
            f'Retrieval: {best.retrieval}\n'
            f'Embedding model: {best.embedding_model}\n'
            f'k: {best.k}\n'
            f'Example order: {best.example_order}\n'
            f'Prompt name: {best.prompt_name}\n\n'
            f'{resolved_prompt}'
        )
    path.write_text('\n\n---\n\n'.join(sections) + '\n', encoding='utf-8')


def _build_conditions(
        target: Column,
        methods: list[str],
        embedding_models: list[dict[str, Any]],
        language_models: list[dict[str, Any]],
        k_values: list[int],
        example_orders: list[str],
        prompt_templates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build the full cross-product of configured experiment conditions."""

    conditions: list[dict[str, Any]] = []
    for language_model in language_models:
        language_model_id = language_model['id']
        for method in methods:
            for embedding_model in embedding_models:
                embedding_model_id = embedding_model['id']
                for k in k_values:
                    for example_order in example_orders:
                        for prompt_name, master_prompt in prompt_templates.items():
                            condition = (
                                f'{target} | model={language_model_id} | '
                                f'{method} | embedding={embedding_model_id} | '
                                f'k={k} | {example_order} | {prompt_name}'
                            )
                            conditions.append(
                                {
                                    'condition': condition,
                                    'retrieval': method,
                                    'embedding_model': embedding_model_id,
                                    'k': k,
                                    'example_order': example_order,
                                    'prompt_name': prompt_name,
                                    'master_prompt': master_prompt,
                                    'model': language_model_id,
                                }
                            )
    return conditions


def run_experiment(
        config_source: str | Path | Mapping[str, Any],
        project_root: str | Path = '.',
        progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the complete experiment and return tables plus output paths."""

    config = load_config(config_source)
    validate_config(config)
    root = Path(project_root).resolve()
    defaults = config['defaults']
    seed = defaults['seed']
    target, audit_column, _, labels = task_settings(config)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_settings = config['model']
    language_models = [
        dict(settings)
        for settings in model_settings['language_models']
    ]
    device = choose_device(model_settings['device'])
    progress(f'Using device: {device}')
    progress(
        f'Holding out {target}; model input is hard_text + {audit_column}'
    )
    train, validation, test, dataset_counts = load_data(config, root)
    progress(
        f'Loaded {len(train)} train, {len(validation)} validation, '
        f'and {len(test)} test biographies'
    )

    methods = list(config['retrieval']['methods'])
    evaluation_rows = validation + test
    embedding_models = [
        dict(settings)
        for settings in config['retrieval']['embedding_models']
    ]
    semantic_resources: dict[str, SemanticResource] = {}
    for model_number, embedding_model in enumerate(embedding_models, start=1):
        embedding_model_id = embedding_model['id']
        progress(
            f'Preparing embedding model '
            f'[{model_number}/{len(embedding_models)}]: '
            f'{embedding_model_id}'
        )
        semantic_table, evaluation_vectors = prepare_semantic_retrieval(
            train,
            evaluation_rows,
            embedding_model,
            device,
            root / config['retrieval']['lancedb_path'],
            progress,
        )
        semantic_resources[embedding_model_id] = {
            'table': semantic_table,
            'validation_vectors': evaluation_vectors[:len(validation)],
            'test_vectors': evaluation_vectors[len(validation):],
        }

    k_values = list(config['retrieval']['k_values'])
    example_orders = list(config['retrieval']['example_orders'])
    prompt_templates = dict(config['prompt_templates'])
    training_cells = tuple(sorted({
        (
            row[Column.PROFESSION],
            row[Column.GENDER],
        )
        for row in train
    }))
    conditions = _build_conditions(
        target,
        methods,
        embedding_models,
        language_models,
        k_values,
        example_orders,
        prompt_templates,
    )

    retrieval_cache: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}

    def generate_condition_predictions(
            setting: Mapping[str, Any],
            queries: list[dict[str, Any]],
            evaluation_split: str,
            tokenizer: PreTrainedTokenizerBase,
            language_model: PreTrainedModel,
    ) -> list[dict[str, Any]]:
        """Predict every row for one prompt condition and one data split."""

        rows: list[dict[str, Any]] = []
        retrieval_method = setting['retrieval']
        embedding_model_id = setting['embedding_model']
        semantic_resource = semantic_resources[embedding_model_id]
        semantic_table = semantic_resource['table']
        query_vectors = semantic_resource[
            f'{evaluation_split}_vectors'
        ]
        for query_index, query in enumerate(queries):
            query_vector = query_vectors[query_index]
            retrieval_key = (
                retrieval_method,
                embedding_model_id,
                query[Column.ID],
            )
            if retrieval_key not in retrieval_cache:
                retrieval_cache[retrieval_key] = retrieve_examples(
                    retrieval_method,
                    query_vector,
                    semantic_table,
                    max(k_values),
                    training_cells,
                )
            examples = retrieval_cache[retrieval_key][:setting['k']]
            examples = order_examples(
                examples,
                setting['example_order'],
                seed,
            )
            messages = build_prompt(
                query,
                examples,
                target,
                labels,
                setting['master_prompt'],
            )
            predicted_label, label_scores = score_allowed_labels(
                messages,
                labels,
                tokenizer,
                language_model,
                device,
            )
            rows.append(
                {
                    'condition': setting['condition'],
                    'evaluation_split': evaluation_split,
                    'target': target,
                    'audit_column': audit_column,
                    'query_id': query[Column.ID],
                    Column.HARD_TEXT.value: query[Column.HARD_TEXT],
                    Column.PROFESSION.value: query[Column.PROFESSION],
                    Column.GENDER.value: query[Column.GENDER],
                    'true_label': query[target],
                    'audit_group': query[audit_column],
                    'predicted_label': predicted_label,
                    'retrieval': retrieval_method,
                    'embedding_model': embedding_model_id,
                    'k': setting['k'],
                    'examples_used': len(examples),
                    'example_order': setting['example_order'],
                    'prompt_name': setting['prompt_name'],
                    'master_prompt': setting['master_prompt'],
                    'model': setting['model'],
                    'device': device,
                    'seed': seed,
                    'example_ids': json.dumps(
                        [example[Column.ID] for example in examples]
                    ),
                    'example_professions': json.dumps(
                        [
                            example[Column.PROFESSION]
                            for example in examples
                        ]
                    ),
                    'example_genders': json.dumps(
                        [example[Column.GENDER] for example in examples]
                    ),
                    'example_scores': json.dumps(
                        [example['retrieval_score'] for example in examples]
                    ),
                    'prompt': json.dumps(messages, ensure_ascii=False),
                    'label_scores': json.dumps(label_scores, sort_keys=True),
                }
            )
        return rows

    validation_prediction_rows: list[dict[str, Any]] = []
    condition_number = 0
    for model_number, language_model_settings in enumerate(
            language_models, start=1
    ):
        language_model_id = language_model_settings['id']
        progress(
            f'Loading language model [{model_number}/{len(language_models)}]: '
            f'{language_model_id}'
        )
        tokenizer, language_model = load_llm(
            language_model_id,
            language_model_settings['revision'],
            device,
            language_model_settings['dtype'],
        )
        try:
            model_conditions = [
                setting
                for setting in conditions
                if setting['model'] == language_model_id
            ]
            for setting in model_conditions:
                condition_number += 1
                progress(
                    f'[{condition_number}/{len(conditions)}] validation: '
                    f'{setting['condition']}'
                )
                validation_prediction_rows.extend(
                    generate_condition_predictions(
                        setting,
                        validation,
                        'validation',
                        tokenizer,
                        language_model,
                    )
                )
        finally:
            del tokenizer, language_model
            clear_model_memory(device)

    validation_predictions = pd.DataFrame(validation_prediction_rows)
    (
        validation_results,
        validation_class_metrics,
        validation_confusion,
        validation_group_metrics,
        validation_fairness_metrics,
    ) = calculate_metrics(validation_predictions, labels)
    validation_results = rank_results(
        validation_results,
        defaults['ranking_metric'],
        defaults['ranking_direction'],
    )
    validation_results.insert(
        2, 'selected_for_test', validation_results['rank'].eq(1)
    )
    selected_validation = validation_results.loc[
        validation_results['selected_for_test']
    ].copy()

    conditions_by_name = {
        setting['condition']: setting for setting in conditions
    }
    selected_by_model = {
        row['model']: row
        for row in selected_validation.to_dict('records')
    }
    test_prediction_rows: list[dict[str, Any]] = []
    for model_number, language_model_settings in enumerate(
            language_models, start=1
    ):
        language_model_id = language_model_settings['id']
        best_validation = selected_by_model[language_model_id]
        selected_setting = conditions_by_name[best_validation['condition']]
        progress(
            f'Loading selected language model '
            f'[{model_number}/{len(language_models)}]: {language_model_id}'
        )
        tokenizer, language_model = load_llm(
            language_model_id,
            language_model_settings['revision'],
            device,
            language_model_settings['dtype'],
        )
        try:
            progress(f'Final test: {selected_setting['condition']}')
            test_prediction_rows.extend(
                generate_condition_predictions(
                    selected_setting,
                    test,
                    'test',
                    tokenizer,
                    language_model,
                )
            )
        finally:
            del tokenizer, language_model
            clear_model_memory(device)
    test_predictions = pd.DataFrame(test_prediction_rows)
    (
        results,
        test_class_metrics,
        test_confusion,
        test_group_metrics,
        test_fairness_metrics,
    ) = calculate_metrics(test_predictions, labels)
    results.insert(0, 'selected_on_validation_rank', 1)
    results.insert(
        1,
        'validation_selection_score',
        results['model'].map(
            selected_validation.set_index('model')[defaults['ranking_metric']]
        ),
    )

    predictions = pd.concat(
        [validation_predictions, test_predictions], ignore_index=True
    )
    class_metrics = pd.concat(
        [validation_class_metrics, test_class_metrics], ignore_index=True
    )
    confusion_matrix = pd.concat(
        [validation_confusion, test_confusion], ignore_index=True
    )
    group_metrics = pd.concat(
        [validation_group_metrics, test_group_metrics], ignore_index=True
    )
    fairness_metrics = pd.concat(
        [validation_fairness_metrics, test_fairness_metrics], ignore_index=True
    )

    run_dir = (
            root
            / defaults['output_dir']
            / datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
    )
    run_dir.mkdir(parents=True)
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    validation_results.to_csv(run_dir / 'validation_results.csv', index=False)
    results.to_csv(run_dir / 'results.csv', index=False)
    class_metrics.to_csv(run_dir / 'class_metrics.csv', index=False)
    confusion_matrix.to_csv(run_dir / 'confusion_matrix.csv', index=False)
    group_metrics.to_csv(run_dir / 'group_metrics.csv', index=False)
    fairness_metrics.to_csv(run_dir / 'fairness_metrics.csv', index=False)
    dataset_counts.to_csv(run_dir / 'dataset_counts.csv', index=False)
    with (run_dir / 'config_used.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    plot_path = run_dir / 'results.png'
    _plot_results(validation_results, plot_path)
    best_prompts_path = run_dir / 'best_prompts.txt'
    _write_best_prompts(
        best_prompts_path,
        selected_validation,
        prompt_templates,
        labels,
        defaults['ranking_metric'],
        defaults['ranking_direction'],
        results,
    )

    progress(f'Finished: {run_dir}')
    return {
        'run_dir': run_dir,
        'predictions': predictions,
        'validation_results': validation_results,
        'results': results,
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix,
        'group_metrics': group_metrics,
        'fairness_metrics': fairness_metrics,
        'dataset_counts': dataset_counts,
        'plot': plot_path,
        'best_prompts': best_prompts_path,
    }
