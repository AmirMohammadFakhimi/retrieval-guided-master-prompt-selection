from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

import configuration
import dataset
import embeddings
import modeling

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
STEPS_BY_MODEL = {
    'Qwen/Qwen3-Embedding-8B': 2,
    'BAAI/bge-large-en-v1.5': 5,
}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = configuration.load_config(
        project_root / 'configs' / 'validation.yaml',
        project_root,
    )
    device = modeling.choose_device(config['inference']['device'])

    source_rows = dataset.load_source_rows(config, project_root)
    length_sorted_training_rows = [row for row in source_rows if row[dataset.Column.SPLIT] == 'train']
    length_sorted_training_rows.sort(key=lambda row: len(row[dataset.Column.HARD_TEXT]), reverse=True)
    length_sorted_training_texts = [row[dataset.Column.HARD_TEXT] for row in length_sorted_training_rows]
    measurements = []

    for embedding_model in config['retrieval']['embedding_models']:
        model_id = embedding_model['id']
        steps = STEPS_BY_MODEL[model_id]
        encoder = embeddings._load_embedding_encoder(embedding_model, device)
        embeddings._warn_about_embedding_input_truncation(
            encoder,
            length_sorted_training_rows[:steps * embeddings.LANCEDB_INGEST_BATCH_SIZE],
            model_id,
            embedding_model['max_sequence_length'],
            'benchmark training-document',
        )

        for batch_size in BATCH_SIZES:
            try:
                # Warm up this batch shape before measuring it.
                encoder.encode(
                    length_sorted_training_texts[:batch_size],
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                for step in range(1, steps + 1):
                    start = (step - 1) * embeddings.LANCEDB_INGEST_BATCH_SIZE
                    texts = length_sorted_training_texts[start:start + embeddings.LANCEDB_INGEST_BATCH_SIZE]

                    started = perf_counter()
                    encoder.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    seconds = perf_counter() - started
                    row_count = len(texts)
                    rows_per_second = row_count / seconds

                    measurements.append({
                        'embedding_model': model_id,
                        'batch_size': batch_size,
                        'step': step,
                        'row_count': row_count,
                        'seconds': seconds,
                        'rows_per_second': rows_per_second,
                    })

                    timestamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')
                    print(
                        f'[{timestamp}] {model_id} | batch_size={batch_size} | '
                        f'step={step}/{steps} | {rows_per_second:.2f} rows/s'
                    )
            except RuntimeError as error:
                timestamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')
                print(f'[{timestamp}] {model_id} | batch_size={batch_size} failed: {error}')
                break

        del encoder
        modeling.clear_model_memory(device)

    measurements = pd.DataFrame(measurements)
    measurements_output_path = project_root / 'results' / 'embedding_batch_size_benchmark_measurements.csv'
    measurements_output_path.parent.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(measurements_output_path, index=False)

    summary = (
        measurements
        .groupby(['embedding_model', 'batch_size'], as_index=False)
        .agg(row_count=('row_count', 'sum'), seconds=('seconds', 'sum'))
    )
    summary['rows_per_second'] = summary['row_count'] / summary['seconds']
    summary_output_path = project_root / 'results' / 'embedding_batch_size_benchmark_summary.csv'
    summary.to_csv(summary_output_path, index=False)

    print('\nAggregate speed')
    print(summary.to_string(index=False))
    print(f'\nSaved measurements to {measurements_output_path}')
    print(f'Saved aggregate summary to {summary_output_path}')


if __name__ == '__main__':
    main()
