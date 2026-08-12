from pathlib import Path
from time import perf_counter

import pandas as pd

import configuration
import dataset
import modeling


BATCH_SIZES = (2, 4, 8, 16, 32, 64, 128, 256)
STEPS_BY_MODEL = {
    'Qwen/Qwen3-Embedding-8B': 2,
    'BAAI/bge-large-en-v1.5': 5,
}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = configuration.load_config(project_root / 'config.yaml')
    device = modeling.choose_device(config['inference']['device'])

    source_rows = dataset.load_source_rows(config, project_root)
    training_texts = [
        row[dataset.Column.HARD_TEXT]
        for row in source_rows
        if row[dataset.Column.SPLIT] == 'train'
    ]
    measurements = []

    for embedding_model in config['retrieval']['embedding_models']:
        model_id = embedding_model['id']
        steps = STEPS_BY_MODEL[model_id]
        encoder = modeling._load_embedding_encoder(embedding_model, device)

        # Do not count one-time model initialization in the first measurement.
        encoder.encode(training_texts[:2], batch_size=2, show_progress_bar=False)

        for batch_size in BATCH_SIZES:
            try:
                for step in range(1, steps + 1):
                    start = (step - 1) * modeling.LANCEDB_INGEST_BATCH_SIZE
                    texts = training_texts[start : start + modeling.LANCEDB_INGEST_BATCH_SIZE]

                    started = perf_counter()
                    encoder.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=True,
                    )
                    seconds = perf_counter() - started
                    rows_per_second = len(texts) / seconds

                    measurements.append({
                            'embedding_model': model_id,
                            'batch_size': batch_size,
                            'step': step,
                            'rows_per_second': rows_per_second,
                        })

                    print(f'{model_id} | batch_size={batch_size} | step={step}/{steps} | {rows_per_second:.2f} rows/s')
            except RuntimeError as error:
                print(f'{model_id} | batch_size={batch_size} failed: {error}')
                break

        del encoder
        modeling.clear_model_memory(device)

    measurements = pd.DataFrame(measurements)
    output_path = project_root / 'results' / 'embedding_batch_size_benchmark.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(output_path, index=False)

    report = measurements.groupby(['embedding_model', 'batch_size'], as_index=False)['rows_per_second'].mean()

    print('\nAverage speed')
    print(report.to_string(index=False))
    print(f'\nSaved measurements to {output_path}')


if __name__ == '__main__':
    main()
