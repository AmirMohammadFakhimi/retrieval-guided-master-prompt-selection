import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset
from ftfy import fix_text
from tqdm.auto import tqdm

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


class Column(StrEnum):
    """Canonical Bias-in-Bios dataset columns."""

    ID = 'id'
    SPLIT = 'split'
    HARD_TEXT = 'hard_text'
    PROFESSION = 'profession'
    GENDER = 'gender'


TARGET_TO_AUDIT_COLUMN = {
    Column.PROFESSION: Column.GENDER,
    Column.GENDER: Column.PROFESSION,
}

INVISIBLE_PATTERN = re.compile(r'[\u00ad\u200b-\u200f\u2060\ufeff]')
CONTROL_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
WHITESPACE_PATTERN = re.compile(r'\s+')
SPACE_BEFORE_PUNCTUATION_PATTERN = re.compile(r'\s+([,.;:!?])')
HTML_MARKUP_PATTERN = re.compile(
    r'<!--.*?-->|<!doctype\s+[^>]*>|'
    r'</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?/?>',
    re.IGNORECASE | re.DOTALL,
)


def task_settings(config: dict[str, Any]) -> tuple[Column, Column, list[str], list[str]]:
    """Return target, audit column, professions, and target labels."""

    try:
        target = Column(config['defaults']['target'])
        audit_column = TARGET_TO_AUDIT_COLUMN[target]
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
    return target, audit_column, professions, target_labels


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
                Column.ID: row[Column.ID],
                Column.SPLIT: row[Column.SPLIT],
                Column.HARD_TEXT: row[Column.HARD_TEXT],
                Column.PROFESSION: row[Column.PROFESSION],
                Column.GENDER: row[Column.GENDER],
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
    shuffle_seed = dataset_config['shuffle_seed']
    rows: list[dict[str, Any]] = []

    for split_name in ('train', 'dev', 'test'):
        split_data = load_dataset(dataset_config['hub_id'], split=split_name).shuffle(seed=shuffle_seed)

        if len(split_data) == 0:
            raise ValueError(f'Bias in Bios {split_name} split is empty')
        for index, row in enumerate(split_data):
            rows.append({
                Column.ID: f'{split_name}:{shuffle_seed}:{index}',
                Column.SPLIT: split_name,
                Column.HARD_TEXT: _clean_text(row[Column.HARD_TEXT]),
                Column.PROFESSION: PROFESSIONS[row[Column.PROFESSION]],
                Column.GENDER: GENDERS[row[Column.GENDER]],
            })

    _write_jsonl(destination, rows)
    return rows


def load_data(config: dict[str, Any], project_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Load and normalize all profession-filtered source splits."""

    dataset_config = config['dataset']
    dataset_path = project_root / dataset_config['file']
    if not dataset_path.exists():
        rows = _download_data(config, dataset_path)
    else:
        rows = _read_jsonl(dataset_path)

    _, _, professions, _ = task_settings(config)
    rows = [row for row in rows if row[Column.PROFESSION] in professions]

    split_rows = {
        'train': [{**row, Column.SPLIT: 'train'} for row in rows if row[Column.SPLIT] == 'train'],
        'validation': [{**row, Column.SPLIT: 'validation'} for row in rows if row[Column.SPLIT] == 'dev'],
        'test': [{**row, Column.SPLIT: 'test'} for row in rows if row[Column.SPLIT] == 'test'],
    }
    empty_splits = [name for name, split in split_rows.items() if not split]
    if empty_splits:
        raise ValueError(f'Dataset {dataset_config["hub_id"]} has no matching rows in canonical splits: {empty_splits}')

    return split_rows


def select_run_data(
        config: dict[str, Any],
        split_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Select the capped retrieval pool and configured evaluation rows."""

    evaluation_split = config['defaults']['evaluation_split']
    missing_splits = {'train', evaluation_split} - set(split_rows)
    if missing_splits:
        raise ValueError(f'split_rows is missing required splits: {sorted(missing_splits)}')

    train_size = train_size_limit(config)
    available_train = split_rows['train']
    train = available_train if train_size is None else available_train[:train_size]

    if train_size is not None and len(train) < train_size:
        raise ValueError(
            f'Only {len(available_train)} matching training rows exist; dataset.train_size requested {train_size}'
        )
    if not train:
        raise ValueError('The training demonstration pool is empty')

    example_counts: list[int] = config['retrieval']['example_counts']
    if max(example_counts) > len(train):
        raise ValueError(
            f'Every retrieval.example_counts entry must be <= the available training pool size ({len(train)})'
        )

    _, _, professions, _ = task_settings(config)
    available_evaluation = split_rows[evaluation_split]
    evaluation_cells: dict[tuple[str, str], list[dict[str, Any]]] = {}

    progress_bar = tqdm(total=len(professions) * len(GENDERS), desc='Selecting evaluation cells', unit='cell')
    for profession in professions:
        for gender in GENDERS:
            evaluation_cells[(profession, gender)] = [
                row
                for row in available_evaluation
                if row[Column.PROFESSION] == profession and row[Column.GENDER] == gender
            ]
            progress_bar.update(1)
    progress_bar.close()

    evaluation_per_profession_gender = config['dataset']['evaluation_per_profession_gender']
    if evaluation_per_profession_gender == 'max_balanced':
        empty_cells = [cell for cell, rows in evaluation_cells.items() if not rows]
        if empty_cells:
            raise ValueError(
                f"Dataset {config['dataset']['hub_id']!r} has no {evaluation_split} rows "
                f'for these profession/gender cells: {empty_cells}'
            )
        evaluation_per_cell = min(map(len, evaluation_cells.values()))
    else:
        evaluation_per_cell = evaluation_per_profession_gender

    missing_cells = [cell for cell, rows in evaluation_cells.items() if len(rows) < evaluation_per_cell]
    if missing_cells:
        raise ValueError(
            f"Dataset {config['dataset']['hub_id']!r} lacks enough {evaluation_split} rows "
            f"for these profession/gender cells: {missing_cells}"
        )

    evaluation = [row for rows in evaluation_cells.values() for row in rows[:evaluation_per_cell]]
    return train, evaluation, evaluation_per_cell


def calculate_dataset_counts(config: dict[str, Any], split_rows: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Describe profession-gender composition for the supplied canonical splits."""

    _, _, professions, _ = task_settings(config)
    if not split_rows:
        raise ValueError('split_rows must contain at least one canonical split')
    unknown_splits = set(split_rows) - {'train', 'validation', 'test'}
    if unknown_splits:
        raise ValueError(f'Unknown canonical dataset splits: {sorted(unknown_splits)}')

    split_names = list(split_rows)
    rows = [
        {**row, Column.SPLIT: split_name}
        for split_name, split in split_rows.items()
        for row in split
    ]
    if not rows:
        raise ValueError('Cannot calculate dataset counts from empty split rows')

    counts = (
        pd.DataFrame(rows)
        .groupby([Column.SPLIT, Column.PROFESSION, Column.GENDER], as_index=False)
        .size()
        .rename(columns={'size': 'count'})
    )
    complete_cells = pd.MultiIndex.from_product(
        [split_names, professions, GENDERS],
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

    return counts


def display_column_name(column: Column) -> str:
    """Return a human-readable name for a dataset column used in prompts."""

    return column.replace('_', ' ').title()
