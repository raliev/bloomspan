import os
import subprocess
import time
from pathlib import Path
import shutil
JAVA_CP = "../spmf-jar/spmf.jar:../corpus-miner-java/out"
def benchmark_scaled():
    target_folders = [
        "generated_tests/generate_test_dataset_200f_500wpf",
        "generated_tests/generate_test_dataset_300f_500wpf",
        "generated_tests/generate_test_dataset_400f_500wpf",
        "generated_tests/generate_test_dataset_500f_500wpf",
        "generated_tests/generate_test_dataset_600f_500wpf"
    ]

    for folder in target_folders:
        num_docs = len(os.listdir(folder))
        #sigma = max(2, int(num_docs * 0.1)) # 10% support threshold
        sigma = 3;
        min_l = 3;

        print(f"\n--- Benchmarking Dataset: {folder} ({num_docs} docs, sigma={sigma}, min_l={min_l}) ---")

        # Run matrix: [Implementation] x [Algorithm]
        runs = [
            ("cpp",  "bloomspan",     f"../corpus-miner/corpus_miner {folder} --n {sigma} --ngrams {min_l} --algo bloomspan"),
            ("cpp",  "vmsp",          f"../corpus-miner/corpus_miner {folder} --n {sigma} --ngrams {min_l} --algo vmsp"),
            ("java", "bloomspan",     f"java -cp {JAVA_CP} com.bloomspan.benchmarks../BenchmarkRunner bloomspan {folder} {sigma} {min_l} "),
            ("java", "vmsp",          f"java -cp {JAVA_CP} com.bloomspan.benchmarks.BenchmarkRunner vmsp      {folder} {sigma} {min_l} "),
            ("java", "gst",           f"java -cp {JAVA_CP} com.bloomspan.benchmarks.BenchmarkRunner gst       {folder} {sigma} {min_l} "),
            ("java", "bidecontiguous",f"java -cp {JAVA_CP} com.bloomspan.benchmarks.BenchmarkRunner bidecontiguous {folder} {sigma} {min_l} "),
            ("java", "bidecontiguous",f"java -cp {JAVA_CP} com.bloomspan.benchmarks.BenchmarkRunner bidecontiguousmaximal {folder} {sigma} {min_l} ")
        ]

        for impl, algo, cmd in runs:
            print(f"Executing {impl}-{algo}...")

            # 1. Define target paths within performance_reports
            perf_dir = Path("performance_reports") / folder
            perf_dir.mkdir(parents=True, exist_ok=True)

            results_target = perf_dir / f"{impl}-{algo}-results.txt"
            perf_log_path = perf_dir / f"{impl}-{algo}-perf.txt"

            # 2. Handle output path arguments
            # C++ writes to results_max.csv by default; Java needs the path as the 5th arg
            if impl == "java":
                full_cmd = f"{cmd} {results_target}"
            else:
                full_cmd = cmd
                # Ensure we don't read old C++ results from a previous run
                if os.path.exists("results_max.csv"):
                    os.remove("results_max.csv")

            print(full_cmd)
            start = time.time()
            process = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
            duration = time.time() - start

            # 3. Save the performance/console log
            with open(perf_log_path, "w") as f:
                f.write(f"Command: {full_cmd}\n")
                f.write(f"Total Wall Time: {duration:.4f}s\n")
                f.write("--- STDOUT ---\n")
                f.write(process.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(process.stderr)

            # 4. Finalize the results file
            if impl == "cpp":
                if os.path.exists("results_max.csv"):
                    # Move the C++ default output to our report directory
                    shutil.move("results_max.csv", results_target)
                else:
                    print(f"Warning: results_max.csv not found for {impl}-{algo}")
            # Note: For Java, the file is already created at results_target by BenchmarkRunner

if __name__ == "__main__":
    benchmark_scaled()