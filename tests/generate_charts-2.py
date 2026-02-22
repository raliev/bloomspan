import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def collect_data():
    base_dir = Path("performance_reports/generated_tests")
    data = []

    # Regex to extract docs and words per file from folder name
    folder_regex = re.compile(r"generate_test_dataset_(\d+)f_(\d+)wpf")
    # Regex to extract Total Wall Time from perf.txt
    time_regex = re.compile(r"Total Wall Time: ([\d.]+)s")

    if not base_dir.exists():
        print(f"Directory {base_dir} not found.")
        return []

    for folder in base_dir.iterdir():
        match = folder_regex.search(folder.name)
        if not match: continue

        docs = int(match.group(1))
        wpf = int(match.group(2))

        for file in folder.glob("*-perf.txt"):
            # Expecting filename format: <impl>-<algo>-perf.txt
            parts = file.name.replace("-perf.txt", "").split("-")
            if len(parts) != 2: continue
            impl, algo = parts

            with open(file, 'r') as f:
                content = f.read()
                time_match = time_regex.search(content)
                if time_match:
                    wall_time = float(time_match.group(1))
                    data.append({
                        'impl': impl,
                        'algo': algo,
                        'docs': docs,
                        'wpf': wpf,
                        'total_words': docs * wpf,
                        'time': wall_time
                    })
    return data

def save_plot(filename):
    plt.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(Path("performance_reports_charts") / filename)
    plt.close()

def plot_by_language(data, impl_name):
    # Filter data for specific implementation
    impl_data = [d for d in data if d['impl'] == impl_name]
    if not impl_data: return

    # 1. Chart: Total Words vs Time
    plt.figure()
    for algo in ["vmsp", "bloomspan", "gst", "bidecontiguous","bidecontiguousmaximal"]:
        subset = sorted([d for d in impl_data if d['algo'] == algo], key=lambda x: x['total_words'])
        if not subset: continue
        plt.plot([d['total_words'] for d in subset], [d['time'] for d in subset],
                 marker='o', label=algo.upper())
    plt.title(f"{impl_name.upper()} Algorithms: Total Words Scalability")
    plt.xlabel("Total Words (Docs * Words Per File)")
    plt.ylabel("Wall Time (seconds)")
    save_plot(f"{impl_name}_total_words.png")

    # 2. Charts: Fixed WPF, Variable Docs
    wpf_values = sorted(list(set(d['wpf'] for d in impl_data)))
    for wpf in wpf_values:
        plt.figure()
        for algo in ["vmsp", "bloomspan", "gst", "bidecontiguous","bidecontiguousmaximal"]:
            subset = sorted([d for d in impl_data if d['algo'] == algo and d['wpf'] == wpf],
                            key=lambda x: x['docs'])
            if len(subset) < 2: continue
            plt.plot([d['docs'] for d in subset], [d['time'] for d in subset],
                     marker='s', label=algo.upper())
        plt.title(f"{impl_name.upper()}: Variable Documents (Fixed {wpf} Words/File)")
        plt.xlabel("Number of Documents")
        plt.ylabel("Wall Time (seconds)")
        save_plot(f"{impl_name}_docs_fixed_wpf_{wpf}.png")

    # 3. Charts: Fixed Docs, Variable WPF
    doc_values = sorted(list(set(d['docs'] for d in impl_data)))
    for docs in doc_values:
        plt.figure()
        for algo in ["vmsp", "bloomspan", "gst", "bidecontiguous","bidecontiguousmaximal"]:
            subset = sorted([d for d in impl_data if d['algo'] == algo and d['docs'] == docs],
                            key=lambda x: x['wpf'])
            if len(subset) < 2: continue
            plt.plot([d['wpf'] for d in subset], [d['time'] for d in subset],
                     marker='^', label=algo.upper())
        plt.title(f"{impl_name.upper()}: Variable Words/File (Fixed {docs} Docs)")
        plt.xlabel("Words Per File")
        plt.ylabel("Wall Time (seconds)")
        save_plot(f"{impl_name}_wpf_fixed_docs_{docs}.png")

if __name__ == "__main__":
    os.makedirs("performance_reports_charts", exist_ok=True)
    benchmark_results = collect_data()

    if benchmark_results:
        plot_by_language(benchmark_results, "cpp")
        plot_by_language(benchmark_results, "java")
        print("Charts generated in 'performance_reports_charts/' folder.")
    else:
        print("No data found to plot.")