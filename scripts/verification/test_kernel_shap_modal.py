# scripts/training/test_kernel_shap_modal.py
import modal

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements('requirements-core.txt')
    .env({'PYTHONPATH': '/root/src'})
    .add_local_dir('src', remote_path='/root/src')
)

app = modal.App('gait-shap-diagnostic', image=image)
volume = modal.Volume.from_name('gait-results', create_if_missing=True)

@app.function(cpu=16, memory=16384, timeout=600, volumes={'/results': volume})
def test_kernel_timing(gait_csv: bytes, partition_json: bytes) -> dict:
    import json, time, os, tempfile, warnings
    import numpy as np
    import polars as pl
    import sys
    sys.path.insert(0, '/root/src')
    warnings.filterwarnings('ignore')

    from features import ALL_FEATURE_COLS
    from explain import get_background_data, subsample_stratified, compute_shap_values

    tmp = tempfile.mkdtemp()
    with open(f'{tmp}/gait.csv', 'wb') as f: f.write(gait_csv)
    with open(f'{tmp}/part.json', 'wb') as f: f.write(partition_json)

    import json as _json
    df = pl.read_csv(f'{tmp}/gait.csv')
    part = _json.loads(partition_json)

    source_pool = df.filter(
        (pl.col('condition') == 'pd') | pl.col('subject_id').is_in(part['control_A'])
    )
    X_source = source_pool.select(ALL_FEATURE_COLS).to_numpy().astype(np.float64)
    y_source = source_pool['label'].to_numpy().astype(int)
    rng = np.random.default_rng(42)
    background = get_background_data(X_source, y_source, k=100, rng=rng)
    X_sub, y_sub, idx_sub = subsample_stratified(X_source, y_source, 50, rng)

    model_path = '/results/models_v2/pd_knn.joblib'
    import joblib
    pipeline = joblib.load(model_path)

    print(f"cpu_count: {os.cpu_count()}", flush=True)
    print("Testing parallel KernelExplainer (50 samples, nsamples=256)...", flush=True)

    t0 = time.time()
    result = compute_shap_values(
        'knn', pipeline, X_sub, y_sub, background, idx_sub,
        pipeline_path=model_path,
    )
    elapsed = time.time() - t0

    # Override nsamples for speed in this test only
    import explain as _explain
    _explain._KERNEL_NSAMPLES = 256

    output = {
        'cpu_count': os.cpu_count(),
        'elapsed_50samples_256nsamples': round(elapsed, 1),
        'shap_shape': list(result['shap_values'].shape),
        'base_value': round(result['base_value'], 4),
        'completeness_error': round(result['completeness_error'], 6),
        'has_nan': bool(np.isnan(result['shap_values']).any()),
        'estimated_1000samples_1024nsamples_min': round(elapsed * (1000/50) * (1024/256) / 60, 1),
    }
    print(json.dumps(output, indent=2), flush=True)
    return output

@app.local_entrypoint()
def main():
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    result = test_kernel_timing.remote(
        gait_csv=(repo / 'data/processed/v2/gait_features_v2.csv').read_bytes(),
        partition_json=(repo / 'data/processed/control_partition.json').read_bytes(),
    )
    print(result)
