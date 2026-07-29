import gc
import hashlib
import inspect
import json
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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
EMBEDDING_DTYPES = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


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

    methods = retrieval['methods']
    if not isinstance(methods, list) or not methods:
        raise ValueError('retrieval.methods must be a non-empty list')
    unknown_methods = sorted(set(methods) - RETRIEVAL_METHODS)
    if unknown_methods:
        raise ValueError(
            f'Unknown retrieval methods {unknown_methods}; expected values '
            f'from {sorted(RETRIEVAL_METHODS)}'
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
    if not str(retrieval['lancedb_path']).strip():
        raise ValueError('retrieval.lancedb_path cannot be empty')
    embedding_models = retrieval['embedding_models']
    if not isinstance(embedding_models, list) or not embedding_models:
        raise ValueError(
            'retrieval.embedding_models must contain at least one model'
        )
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
            raise ValueError(
                f'retrieval.embedding_models[{index}] must be a mapping'
            )
        missing_settings = sorted(
            required_embedding_settings - set(embedding_model)
        )
        if missing_settings:
            raise ValueError(
                f'retrieval.embedding_models[{index}] is missing: '
                f'{missing_settings}'
            )
        model_id = str(embedding_model['id']).strip()
        if not model_id:
            raise ValueError(
                f'retrieval.embedding_models[{index}].id cannot be empty'
            )
        embedding_model_ids.append(model_id)
        for setting in (
                'dimension',
                'max_sequence_length',
                'batch_size',
        ):
            value = embedding_model[setting]
            if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
            ):
                raise ValueError(
                    f'retrieval.embedding_models[{index}].{setting} '
                    'must be a positive integer'
                )
        if embedding_model['dtype'] not in EMBEDDING_DTYPES:
            allowed_dtypes = ', '.join(EMBEDDING_DTYPES)
            raise ValueError(
                f'retrieval.embedding_models[{index}].dtype must be '
                f'one of: {allowed_dtypes}'
            )
        if not str(embedding_model['query_prompt']).strip():
            raise ValueError(
                f'retrieval.embedding_models[{index}].query_prompt '
                'cannot be empty'
            )
    if len(embedding_model_ids) != len(set(embedding_model_ids)):
        raise ValueError(
            'retrieval.embedding_models cannot contain duplicate IDs'
        )

    orders = retrieval['example_orders']
    if not isinstance(orders, list) or not orders:
        raise ValueError('retrieval.example_orders must be a non-empty list')
    unknown_orders = sorted(set(orders) - EXAMPLE_ORDERS)
    if unknown_orders:
        raise ValueError(
            f'Unknown example orders {unknown_orders}; expected values '
            f'from {sorted(EXAMPLE_ORDERS)}'
        )
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
        model_kwargs={'dtype': EMBEDDING_DTYPES[dtype_name]},
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
        return sorted(
            examples,
            key=lambda example: hashlib.sha256(
                f'{seed}:{example[Column.ID]}'.encode()
            ).digest(),
        )

    raise RuntimeError(f'Example order {order!r} is allowed but not implemented')


def display_column_name(column: Column | str) -> str:
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
) -> str:
    """Build master instruction + labeled examples + target-free query."""

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

    example_blocks = []
    for number, example in enumerate(examples, start=1):
        example_blocks.append(
            f'Example {number}:\n'
            f'{render_input(example, target)}\n'
            f'{target_name}: {example[target]}'
        )

    query_block = (
        'Query:\n'
        f'{render_input(query, target)}\n'
        f'{target_name}:'
    )

    sections = [instruction_block]
    if example_blocks:
        sections.append(f'Examples:\n{'\n\n'.join(example_blocks)}')
    sections.append(query_block)

    return '\n\n'.join(sections)


def load_language_model(model_id: str, device: str) -> tuple[Any, Any]:
    """Load one local Hugging Face causal language model."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float32 if device == 'cpu' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def _format_model_input(prompt: str, tokenizer: Any) -> str:
    """Apply the model's chat wrapper before scoring allowed labels."""

    if getattr(tokenizer, 'chat_template', None):
        messages = [{'role': 'user', 'content': prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return prompt


def score_allowed_labels(
        prompt: str,
        labels: list[str],
        tokenizer: Any,
        model: Any,
        device: str,
) -> tuple[str, dict[str, float]]:
    """Choose among allowed labels using mean conditional token log-probability.

    Each canonical label is scored as a continuation of the formatted model
    input. Mean rather than summed log-probability reduces the automatic
    disadvantage of labels that use more tokenizer tokens. Labels are evaluated
    sequentially to keep accelerator memory small.
    """

    if not labels:
        raise ValueError('At least one allowed label is required')

    model_input = _format_model_input(prompt, tokenizer)
    prompt_ids = tokenizer(model_input, return_tensors='pt')['input_ids'][0]
    if prompt_ids.numel() == 0:
        raise ValueError('The formatted prompt produced no tokens')

    try:
        forward_parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        forward_parameters = {}
    supports_limited_logits = 'logits_to_keep' in forward_parameters

    scores: dict[str, float] = {}
    for label in labels:
        # The leading space is part of the candidate continuation after
        # 'Answer (...):' and is applied equally to every allowed label.
        candidate_ids = tokenizer(
            f' {label}',
            add_special_tokens=False,
            return_tensors='pt',
        )['input_ids'][0]
        if candidate_ids.numel() == 0:
            raise ValueError(f'Allowed label {label!r} produced no tokens')

        full_ids = torch.cat((prompt_ids, candidate_ids)).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(full_ids)
        forward_kwargs: dict[str, Any] = {
            'input_ids': full_ids,
            'attention_mask': attention_mask,
            'use_cache': False,
        }

        with torch.inference_mode():
            if supports_limited_logits:
                # The final C+1 positions begin at the logit that predicts the
                # first of C candidate tokens. The last position is unused.
                output = model(
                    **forward_kwargs,
                    logits_to_keep=int(candidate_ids.numel()) + 1,
                )
                candidate_logits = output.logits[
                    0, : int(candidate_ids.numel()), :
                ]
            else:
                output = model(**forward_kwargs)
                start = int(prompt_ids.numel()) - 1
                stop = start + int(candidate_ids.numel())
                candidate_logits = output.logits[0, start:stop, :]

            if candidate_logits.shape[0] != candidate_ids.numel():
                raise RuntimeError(
                    f'Could not align model logits for allowed label {label!r}'
                )
            token_log_probabilities = torch.log_softmax(
                candidate_logits.float(), dim=-1
            )
            candidate_on_device = candidate_ids.to(device)
            selected_token_log_probabilities = token_log_probabilities.gather(
                1, candidate_on_device.unsqueeze(1)
            ).squeeze(1)
            score = float(selected_token_log_probabilities.mean().item())

        if not np.isfinite(score):
            raise RuntimeError(
                f'Model returned a non-finite score for allowed label {label!r}'
            )
        scores[label] = score

    predicted_label = max(labels, key=scores.__getitem__)
    return predicted_label, scores


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _rate_range(values: list[float]) -> float:
    defined = [float(value) for value in values if not pd.isna(value)]
    return float(max(defined) - min(defined)) if len(defined) >= 2 else np.nan


def _mean_defined(values: pd.Series | list[float]) -> float:
    defined = pd.Series(values, dtype='float64').dropna()
    return float(defined.mean()) if not defined.empty else np.nan


def _max_defined(values: pd.Series | list[float]) -> float:
    defined = pd.Series(values, dtype='float64').dropna()
    return float(defined.max()) if not defined.empty else np.nan


def _min_defined(values: pd.Series | list[float]) -> float:
    defined = pd.Series(values, dtype='float64').dropna()
    return float(defined.min()) if not defined.empty else np.nan


def _condition_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {column: frame[column].iloc[0] for column in CONDITION_COLUMNS}


def calculate_metrics(
        predictions: pd.DataFrame,
        labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate hard-label classification and group-disparity metrics."""

    if predictions.empty:
        raise ValueError('No predictions were produced')
    required = {
        'condition',
        'true_label',
        'predicted_label',
        'audit_group',
        *CONDITION_COLUMNS,
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f'Predictions are missing required columns: {missing}')
    if predictions[['true_label', 'audit_group']].isna().any().any():
        raise ValueError('true_label and audit_group cannot be missing')

    result_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []

    for _, frame in predictions.groupby(
            ['evaluation_split', 'condition'], sort=False
    ):
        meta = _condition_metadata(frame)
        truth = frame['true_label'].astype(str)
        unknown_truth = sorted(set(truth) - set(labels))
        if unknown_truth:
            raise ValueError(f'True labels outside the configured classes: {unknown_truth}')
        if frame['predicted_label'].isna().any():
            raise ValueError('predicted_label cannot be missing')
        predicted = frame['predicted_label'].astype(str)
        unknown_predictions = sorted(set(predicted) - set(labels))
        if unknown_predictions:
            raise ValueError(
                'Predictions outside the configured classes: '
                f'{unknown_predictions}'
            )
        evaluated_labels = [label for label in labels if (truth == label).any()]
        if not evaluated_labels:
            raise ValueError('No configured target labels appear in the test set')

        per_class: list[dict[str, Any]] = []
        for label in evaluated_labels:
            true_positive = int(((truth == label) & (predicted == label)).sum())
            false_positive = int(((truth != label) & (predicted == label)).sum())
            false_negative = int(((truth == label) & (predicted != label)).sum())
            true_negative = int(((truth != label) & (predicted != label)).sum())
            support = true_positive + false_negative
            precision = _safe_rate(true_positive, true_positive + false_positive)
            recall = _safe_rate(true_positive, support)
            f1 = _safe_rate(2 * true_positive, 2 * true_positive + false_positive + false_negative)
            specificity = _safe_rate(true_negative, true_negative + false_positive)
            row = {
                **meta,
                'target_class': label,
                'support': support,
                'predicted_count': true_positive + false_positive,
                'tp': true_positive,
                'fp': false_positive,
                'fn': false_negative,
                'tn': true_negative,
                'precision': 0.0 if pd.isna(precision) else precision,
                'recall': recall,
                'f1': 0.0 if pd.isna(f1) else f1,
                'specificity': specificity,
                'false_positive_rate': _safe_rate(
                    false_positive, false_positive + true_negative
                ),
                'false_negative_rate': _safe_rate(
                    false_negative, false_negative + true_positive
                ),
                'negative_predictive_value': _safe_rate(
                    true_negative, true_negative + false_negative
                ),
            }
            per_class.append(row)
            class_rows.append(row)

        class_frame = pd.DataFrame(per_class)
        n = len(frame)
        correct = int((truth == predicted).sum())
        weights = class_frame['support'] / n

        # For single-label classification, each wrong answer contributes one
        # pooled FP and one pooled FN. Thus all three micro scores equal
        # accuracy; they are retained for completeness.
        micro_true_positive = correct
        micro_false_positive = n - correct
        micro_false_negative = n - correct
        micro_precision = _safe_rate(
            micro_true_positive, micro_true_positive + micro_false_positive
        )
        micro_recall = _safe_rate(
            micro_true_positive, micro_true_positive + micro_false_negative
        )
        micro_f1 = _safe_rate(
            2 * micro_true_positive,
            2 * micro_true_positive
            + micro_false_positive
            + micro_false_negative,
        )

        for true_label in evaluated_labels:
            for predicted_label in labels:
                confusion_rows.append(
                    {
                        **meta,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'count': int(
                            ((truth == true_label) & (predicted == predicted_label)).sum()
                        ),
                    }
                )

        audit_groups = list(dict.fromkeys(frame['audit_group'].astype(str)))
        group_accuracy_values: list[float] = []
        condition_fairness: list[dict[str, Any]] = []

        for target_class in evaluated_labels:
            class_group_rows: list[dict[str, Any]] = []
            for audit_group in audit_groups:
                group_frame = frame[frame['audit_group'].astype(str) == audit_group]
                group_truth = group_frame['true_label'].astype(str)
                group_predicted = predicted.loc[group_frame.index]
                actual_positive = group_truth == target_class
                predicted_positive = group_predicted == target_class
                tp = int((actual_positive & predicted_positive).sum())
                fp = int((~actual_positive & predicted_positive).sum())
                fn = int((actual_positive & ~predicted_positive).sum())
                tn = int((~actual_positive & ~predicted_positive).sum())
                group_n = len(group_frame)
                group_accuracy = _safe_rate(
                    int((group_truth == group_predicted).sum()), group_n
                )
                row = {
                    **meta,
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
                    'selection_rate': _safe_rate(tp + fp, group_n),
                    'true_positive_rate': _safe_rate(tp, tp + fn),
                    'false_positive_rate': _safe_rate(fp, fp + tn),
                    'false_negative_rate': _safe_rate(fn, tp + fn),
                    'positive_predictive_value': _safe_rate(tp, tp + fp),
                    'specificity': _safe_rate(tn, tn + fp),
                    'negative_predictive_value': _safe_rate(tn, tn + fn),
                }
                class_group_rows.append(row)
                group_rows.append(row)

            rates = pd.DataFrame(class_group_rows)
            selection_values = rates['selection_rate'].dropna()
            demographic_parity_ratio = (
                float(selection_values.min() / selection_values.max())
                if len(selection_values) >= 2 and selection_values.max() > 0
                else np.nan
            )
            equal_opportunity_difference = _rate_range(
                rates['true_positive_rate'].tolist()
            )
            false_positive_rate_difference = _rate_range(
                rates['false_positive_rate'].tolist()
            )
            equalized_odds_difference = (
                max(equal_opportunity_difference, false_positive_rate_difference)
                if not pd.isna(equal_opportunity_difference)
                   and not pd.isna(false_positive_rate_difference)
                else np.nan
            )
            fairness_row = {
                **meta,
                'target_class': target_class,
                'groups_compared': len(audit_groups),
                'selection_rate_groups_defined': int(
                    rates['selection_rate'].notna().sum()
                ),
                'tpr_groups_defined': int(
                    rates['true_positive_rate'].notna().sum()
                ),
                'fpr_groups_defined': int(
                    rates['false_positive_rate'].notna().sum()
                ),
                'ppv_groups_defined': int(
                    rates['positive_predictive_value'].notna().sum()
                ),
                'demographic_parity_difference': _rate_range(
                    rates['selection_rate'].tolist()
                ),
                'demographic_parity_ratio': demographic_parity_ratio,
                'equal_opportunity_difference': equal_opportunity_difference,
                'false_positive_rate_difference': false_positive_rate_difference,
                'equalized_odds_difference': equalized_odds_difference,
                'predictive_parity_difference': _rate_range(
                    rates['positive_predictive_value'].tolist()
                ),
            }
            fairness_rows.append(fairness_row)
            condition_fairness.append(fairness_row)

        # Group-wide accuracy does not depend on a target class.
        for audit_group in audit_groups:
            group_frame = frame[frame['audit_group'].astype(str) == audit_group]
            group_truth = group_frame['true_label'].astype(str)
            group_predicted = predicted.loc[group_frame.index]
            group_accuracy_values.append(float((group_truth == group_predicted).mean()))

        fairness_frame = pd.DataFrame(condition_fairness)
        result_rows.append(
            {
                **meta,
                'n': n,
                'n_classes': len(evaluated_labels),
                'n_audit_groups': len(audit_groups),
                'accuracy': _safe_rate(correct, n),
                'balanced_accuracy': float(class_frame['recall'].mean()),
                'macro_precision': float(class_frame['precision'].mean()),
                'macro_recall': float(class_frame['recall'].mean()),
                'macro_f1': float(class_frame['f1'].mean()),
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1,
                'weighted_precision': float(
                    (class_frame['precision'] * weights).sum()
                ),
                'weighted_recall': float((class_frame['recall'] * weights).sum()),
                'weighted_f1': float((class_frame['f1'] * weights).sum()),
                'matthews_correlation_coefficient': float(
                    matthews_corrcoef(truth, predicted)
                ),
                'cohen_kappa': float(cohen_kappa_score(truth, predicted)),
                'worst_group_accuracy': (
                    min(group_accuracy_values) if group_accuracy_values else np.nan
                ),
                'group_accuracy_difference': _rate_range(group_accuracy_values),
                'mean_demographic_parity_difference': _mean_defined(
                    fairness_frame['demographic_parity_difference']
                ),
                'max_demographic_parity_difference': _max_defined(
                    fairness_frame['demographic_parity_difference']
                ),
                'mean_demographic_parity_ratio': _mean_defined(
                    fairness_frame['demographic_parity_ratio']
                ),
                'min_demographic_parity_ratio': _min_defined(
                    fairness_frame['demographic_parity_ratio']
                ),
                'mean_equal_opportunity_difference': _mean_defined(
                    fairness_frame['equal_opportunity_difference']
                ),
                'max_equal_opportunity_difference': _max_defined(
                    fairness_frame['equal_opportunity_difference']
                ),
                'mean_false_positive_rate_difference': _mean_defined(
                    fairness_frame['false_positive_rate_difference']
                ),
                'max_false_positive_rate_difference': _max_defined(
                    fairness_frame['false_positive_rate_difference']
                ),
                'mean_equalized_odds_difference': _mean_defined(
                    fairness_frame['equalized_odds_difference']
                ),
                'max_equalized_odds_difference': _max_defined(
                    fairness_frame['equalized_odds_difference']
                ),
                'n_demographic_parity_defined_classes': int(
                    fairness_frame['demographic_parity_difference'].notna().sum()
                ),
                'n_demographic_parity_ratio_defined_classes': int(
                    fairness_frame['demographic_parity_ratio'].notna().sum()
                ),
                'n_equal_opportunity_defined_classes': int(
                    fairness_frame['equal_opportunity_difference'].notna().sum()
                ),
                'n_false_positive_rate_defined_classes': int(
                    fairness_frame['false_positive_rate_difference'].notna().sum()
                ),
                'n_equalized_odds_defined_classes': int(
                    fairness_frame['equalized_odds_difference'].notna().sum()
                ),
                'n_predictive_parity_defined_classes': int(
                    fairness_frame['predictive_parity_difference'].notna().sum()
                ),
                'mean_predictive_parity_difference': _mean_defined(
                    fairness_frame['predictive_parity_difference']
                ),
                'max_predictive_parity_difference': _max_defined(
                    fairness_frame['predictive_parity_difference']
                ),
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
    """Rank prompt configurations by the configured primary metric."""

    if metric not in results.columns:
        available = ', '.join(results.select_dtypes(include='number').columns)
        raise ValueError(
            f'Unknown ranking metric {metric!r}. Numeric result columns: {available}'
        )
    if not pd.api.types.is_numeric_dtype(results[metric]):
        raise ValueError(f'Ranking metric {metric!r} must be numeric')
    if results[metric].notna().sum() == 0:
        raise ValueError(f'Ranking metric {metric!r} is undefined for every condition')
    ranked = results.copy()
    ranked.insert(
        0,
        'rank',
        ranked[metric].rank(
            method='min',
            ascending=direction == 'minimize',
            na_option='bottom',
        ).astype('Int64'),
    )
    ranked.insert(1, 'is_best', ranked['rank'] == 1)
    return ranked.sort_values(['rank', 'condition'], kind='stable').reset_index(drop=True)


def _plot_results(
        validation_results: pd.DataFrame,
        output: Path,
        test_results: pd.DataFrame,
) -> None:
    """Plot validation comparisons and annotate the selected prompt's test score."""

    import matplotlib.pyplot as plt

    plot_frame = validation_results.sort_values(
        'rank', ascending=False
    ).reset_index(drop=True)
    labels = [
        (
            f'{row.retrieval}, embedding={row.embedding_model}, '
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
    final = test_results.iloc[0]
    figure.suptitle(
        f'Validation prompt comparison — target: {plot_frame['target'].iloc[0]}, '
        f'audit groups: {plot_frame['audit_column'].iloc[0]}\n'
        f'Selected prompt final test: accuracy={final['accuracy']:.3f}, '
        f'macro-F1={final['macro_f1']:.3f}, '
        f'max equalized-odds difference='
        f'{final['max_equalized_odds_difference']:.3f}'
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def _write_best_prompt(
        path: Path,
        best: pd.Series,
        prompt_templates: Mapping[str, Any],
        labels: list[str],
        ranking_metric: str,
        ranking_direction: str,
        final_result: pd.Series,
) -> None:
    """Save the validation-selected prompt and final-test score in plain text."""

    resolved_prompt = str(prompt_templates[best['prompt_name']]).format(
        target=display_column_name(best['target']),
        other_column=display_column_name(best['audit_column']),
        labels=', '.join(labels),
    )
    text = (
        f'Selected on validation metric: {ranking_metric} ({ranking_direction})\n'
        f'Validation score: {best[ranking_metric]}\n'
        f'Final test score for the same metric: {final_result[ranking_metric]}\n'
        f'Final test accuracy: {final_result['accuracy']}\n'
        f'Final test macro-F1: {final_result['macro_f1']}\n'
        f'Target: {best['target']}\n'
        f'Retrieval: {best['retrieval']}\n'
        f'Embedding model: {best['embedding_model']}\n'
        f'k: {best['k']}\n'
        f'Example order: {best['example_order']}\n'
        f'Prompt name: {best['prompt_name']}\n'
        f'Model: {best['model']}\n\n'
        'Resolved master prompt:\n'
        f'{resolved_prompt}\n\n'
        'The complete prompt also contains the retrieved examples, allowed labels, '
        'and each evaluation query; those are saved in predictions.csv.\n'
    )
    path.write_text(text, encoding='utf-8')


def _build_conditions(
        target: Column,
        methods: list[str],
        embedding_models: list[dict[str, Any]],
        k_values: list[int],
        example_orders: list[str],
        prompt_templates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Cross every semantic retrieval method with every embedding model."""

    conditions: list[dict[str, Any]] = []
    for method in methods:
        for embedding_model in embedding_models:
            embedding_model_id = str(embedding_model['id'])
            for k in k_values:
                for example_order in example_orders:
                    for prompt_name, master_prompt in prompt_templates.items():
                        condition = (
                            f'{target} | {method} | '
                            f'embedding={embedding_model_id} | k={k} | '
                            f'{example_order} | {prompt_name}'
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
    seed = int(defaults['seed'])
    target, audit_column, _, labels = task_settings(config)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = choose_device(config['model']['device'])
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
        embedding_model_id = str(embedding_model['id'])
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

    tokenizer, model = load_language_model(config['model']['id'], device)
    k_values = [int(value) for value in config['retrieval']['k_values']]
    example_orders = list(config['retrieval']['example_orders'])
    prompt_templates = dict(config['prompt_templates'])
    training_cells = tuple(sorted({
        (
            str(row[Column.PROFESSION]),
            str(row[Column.GENDER]),
        )
        for row in train
    }))
    conditions = _build_conditions(
        target,
        methods,
        embedding_models,
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
    ) -> list[dict[str, Any]]:
        """Predict every row for one prompt condition and one data split."""

        rows: list[dict[str, Any]] = []
        retrieval_method = str(setting['retrieval'])
        embedding_model_id = str(setting['embedding_model'])
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
                str(query[Column.ID]),
            )
            if retrieval_key not in retrieval_cache:
                retrieval_cache[retrieval_key] = retrieve_examples(
                    retrieval_method,
                    query_vector,
                    semantic_table,
                    max(k_values),
                    training_cells,
                )
            examples = retrieval_cache[retrieval_key][:int(setting['k'])]
            examples = order_examples(
                examples,
                str(setting['example_order']),
                seed,
            )
            prompt = build_prompt(
                query,
                examples,
                target,
                labels,
                str(setting['master_prompt']),
            )
            predicted_label, label_scores = score_allowed_labels(
                prompt,
                labels,
                tokenizer,
                model,
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
                    'k': int(setting['k']),
                    'examples_used': len(examples),
                    'example_order': setting['example_order'],
                    'prompt_name': setting['prompt_name'],
                    'master_prompt': setting['master_prompt'],
                    'model': config['model']['id'],
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
                    'prompt': prompt,
                    'label_scores': json.dumps(label_scores, sort_keys=True),
                }
            )
        return rows

    validation_prediction_rows: list[dict[str, Any]] = []
    for condition_number, setting in enumerate(conditions, start=1):
        progress(
            f'[{condition_number}/{len(conditions)}] validation: '
            f'{setting['condition']}'
        )
        validation_prediction_rows.extend(
            generate_condition_predictions(
                setting,
                validation,
                'validation',
            )
        )

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
        str(defaults['ranking_metric']),
        str(defaults['ranking_direction']),
    )
    validation_results.insert(2, 'selected_for_test', False)
    validation_results.loc[0, 'selected_for_test'] = True
    best_validation = validation_results.iloc[0]
    selected_setting = next(
        setting
        for setting in conditions
        if setting['condition'] == best_validation['condition']
    )

    progress(f'Final test: {selected_setting['condition']}')
    test_predictions = pd.DataFrame(
        generate_condition_predictions(
            selected_setting,
            test,
            'test',
        )
    )
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
        best_validation[str(defaults['ranking_metric'])],
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
            / str(defaults['output_dir'])
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
    _plot_results(validation_results, plot_path, results)
    best_prompt_path = run_dir / 'best_prompt.txt'
    _write_best_prompt(
        best_prompt_path,
        best_validation,
        prompt_templates,
        labels,
        str(defaults['ranking_metric']),
        str(defaults['ranking_direction']),
        results.iloc[0],
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
        'best_prompt': best_prompt_path,
    }
