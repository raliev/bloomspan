# Define the test case and sigma
CASE="test-gen/my_tests/01_max_check"
SIGMA=3

echo "--- Testing VMSP (The Gold Standard for Maximal) ---"
./corpus_miner $CASE --n $SIGMA --algo vmsp --min_l 1
echo "=============================="
cat results_max.csv
echo "=============================="

echo "--- Testing Your Algorithm (BloomSpan) ---"
# Note: ngrams 1 allows it to find any length phrase
./corpus_miner $CASE --n $SIGMA --algo bloomspan --ngrams 1
echo "=============================="
cat results_max.csv
echo "=============================="
