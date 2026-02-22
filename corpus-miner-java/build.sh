#!/bin/bash
set -e

# Change to the script's directory (spmf-extensions)
cd "$(dirname "$0")"

echo "Compiling BenchmarkRunner and Extension Algorithms..."

# Clean previous build
rm -rf out
mkdir -p out

# Compile all .java files using the root spmf.jar as the classpath
javac -Xlint:none -cp "../spmf-jar/spmf.jar:src" \
    src/ca/pfv/spmf/algorithms/sequentialpatterns/spam/*.java \
    src/ca/pfv/spmf/algorithms/sequentialpatterns/BIDE_and_prefixspan/*.java \
    src/com/bloomspan/benchmarks/BenchmarkRunner.java \
    -d out

echo "Building extensions.jar..."
jar cf extensions.jar -C out .

echo "Build successful! Created extensions.jar."
echo ""
echo "To run the benchmark:"
echo "java -cp \"../spmf.jar:extensions.jar\" BenchmarkRunner <algo> <folder> <minSupportAbs> <minLen> <outputCsv>"
