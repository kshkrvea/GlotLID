import time
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from customlid import CustomLID
import tqdm

model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin", cache_dir=None)

dataset = load_dataset("cis-lmu/udhr-lid", split="test")
test_texts = [row["sentence"] for row in dataset]

N_RUNS = 10
N_SPEED = min(5000, len(test_texts))
K = 1

speed_texts = (test_texts * (N_SPEED // len(test_texts) + 1))[:N_SPEED]

pairs = [
    ("before",     "optimized_before"),
    ("after",      "optimized_after"),
]

report_lines = []

for mode_orig, mode_opt in pairs:
    model_original = CustomLID(model_path, mode=mode_orig)
    model_optimized = CustomLID(model_path, mode=mode_opt)

    max_diff = 0.0
    for text in test_texts:
        _, probs_orig = model_original.predict(text, K)
        _, probs_opt = model_optimized.predict(text, K)
        max_diff = max(max_diff, float(np.max(np.abs(probs_orig - probs_opt))))

    for text in speed_texts[:10]:
        model_original.predict(text, K)
        model_optimized.predict(text, K)

    elapsed_orig_runs = []
    elapsed_opt_runs = []

    for _ in tqdm.tqdm(range(N_RUNS), desc=f"Benchmarking {mode_orig}/{mode_opt}"):
        start = time.perf_counter()
        for text in speed_texts:
            model_original.predict(text, K)
        elapsed_orig_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        for text in speed_texts:
            model_optimized.predict(text, K)
        elapsed_opt_runs.append(time.perf_counter() - start)

    elapsed_orig = np.mean(elapsed_orig_runs)
    elapsed_opt = np.mean(elapsed_opt_runs)

    samples_per_sec_orig = N_SPEED / elapsed_orig
    samples_per_sec_opt = N_SPEED / elapsed_opt
    seconds_per_sample_orig = elapsed_orig / N_SPEED
    seconds_per_sample_opt = elapsed_opt / N_SPEED
    speedup = (elapsed_orig - elapsed_opt) / elapsed_orig * 100

    report_lines.append(
        f"[{mode_orig} vs {mode_opt}]\n"
        f"Max Absolute Difference: {max_diff:.2e}\n"
        f"Speed Original: {samples_per_sec_orig:.1f} samples/sec, {seconds_per_sample_orig:.6f} sec/sample\n"
        f"Speed Optimized: {samples_per_sec_opt:.1f} samples/sec, {seconds_per_sample_opt:.6f} sec/sample\n"
        f"Total Speedup: {speedup:.1f}%"
    )

report = "\n\n".join(report_lines)

with open("report.txt", "w") as f:
    f.write(report + "\n")

print(report)
