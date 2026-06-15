package ca.pfv.spmf.algorithms.sequentialpatterns.dfi;

import java.io.*;
import java.util.*;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.AlgoBloomSpan;
import ca.pfv.spmf.tools.MemoryLogger;

/**
 * AlgoDFI – Deferred Frequency Index for maximal frequent contiguous phrases.
 *
 * Inspired by Weese & Schulz (2008) "Efficient string mining under constraints
 * via the deferred frequency index". Uses suffix array + LCP array with a
 * top-down traversal of the virtual LCP tree and early subtree pruning:
 * when a node's document-frequency is below the threshold its entire subtree
 * is skipped immediately, without examining any of its descendants.
 *
 * Key algorithmic difference from AlgoFHK (bottom-up stack traversal):
 *   - DFI goes top-down; as soon as df(node) < minSup the whole subtree
 *     is abandoned — no scanning of individual leaves in that range.
 *   - Right-maximality is determined by inspecting children inline during
 *     the same linear scan that computes maxChildDf.
 *   - An O(n log n) sparse table enables O(1) RMQ to find each node's
 *     string-depth from its SA range.
 *
 * Output is identical to AlgoFHK and AlgoBloomSpan.
 */
public class AlgoDFI {

    private int minSupportAbs;
    private int minLen = 1;

    // Corpus
    private int numDocs;
    private int[] text;          // concatenated token IDs + per-doc sentinels + global sentinel
    private int n;               // text length
    private int alphabetSize;
    private int vocabSize;
    private int[] docId;         // docId[pos] = document that owns text[pos]
    private int[] docStart;      // docStart[d] = first position in text[] for doc d

    // Index structures
    private int[] sa;            // suffix array
    private int[] lcp;           // lcp[i] = LCP(sa[i-1], sa[i]); lcp[0] = 0
    private int[] pred;          // pred[i] = previous SA-rank of same doc, or -1

    // Sparse table for O(1) RMQ on lcp[]
    private int[][] rmqTable;
    private int[]   rmqLog;

    public void setMinL(int minL) {
        this.minLen = minL;
    }

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public List<AlgoBloomSpan.Phrase> runAlgorithm(String input, String output,
                                                    double minsupRel) throws IOException {
        MemoryLogger.getInstance().reset();

        loadCorpus(input);
        MemoryLogger.getInstance().checkMemory();

        this.minSupportAbs = (int) Math.ceil(minsupRel * numDocs);
        if (this.minSupportAbs == 0) this.minSupportAbs = 1;

        buildSuffixArray();
        MemoryLogger.getInstance().checkMemory();
        buildLCP();
        buildPred();
        buildRMQ();
        MemoryLogger.getInstance().checkMemory();

        List<AlgoBloomSpan.Phrase> result = mineMCFPs();
        MemoryLogger.getInstance().checkMemory();
        System.out.println("MAXMEMORY=" + MemoryLogger.getInstance().getMaxMemory());
        return result;
    }

    // -------------------------------------------------------------------------
    // Corpus loading  (identical to AlgoFHK)
    // -------------------------------------------------------------------------

    private void loadCorpus(String input) throws IOException {
        List<int[]> perDocTokens = new ArrayList<>();
        int maxId = 0;
        long totalTokens = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(input))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split("\\s+");
                int kept = 0;
                int[] ids = new int[parts.length];
                for (String pt : parts) {
                    if (pt.isEmpty() || pt.equals("-1") || pt.equals("-2")) continue;
                    int id = Integer.parseInt(pt);
                    if (id > maxId) maxId = id;
                    ids[kept++] = id;
                }
                ids = Arrays.copyOf(ids, kept);
                perDocTokens.add(ids);
                totalTokens += kept;
            }
        }

        numDocs     = perDocTokens.size();
        vocabSize   = maxId;
        int tokenCount = (int) Math.min(totalTokens, Integer.MAX_VALUE);

        // Layout: doc0-tokens sep0  doc1-tokens sep1  ...  global-sentinel
        n      = tokenCount + numDocs + 1;
        text   = new int[n];
        docId  = new int[n];
        docStart = new int[numDocs];

        int pos = 0;
        for (int d = 0; d < numDocs; d++) {
            int[] ids = perDocTokens.get(d);
            docStart[d] = pos;
            for (int t : ids) {
                text[pos]  = t;
                docId[pos] = d;
                pos++;
            }
            text[pos]  = vocabSize + 1 + d;   // unique separator
            docId[pos] = d;
            pos++;
        }
        text[pos]  = 0;                        // global sentinel (smallest value)
        docId[pos] = numDocs;

        alphabetSize = vocabSize + 1 + numDocs;
    }

    // -------------------------------------------------------------------------
    // Suffix array via SA-IS  (identical to AlgoFHK)
    // -------------------------------------------------------------------------

    private void buildSuffixArray() {
        sa = new int[n];
        SAIS.build(text, sa, n, alphabetSize);
    }

    // -------------------------------------------------------------------------
    // LCP array via Kasai's algorithm  (identical to AlgoFHK)
    // -------------------------------------------------------------------------

    private void buildLCP() {
        lcp  = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int h = 0;
        for (int i = 0; i < n; i++) {
            int r = rank[i];
            if (r > 0) {
                int j = sa[r - 1];
                while (i + h < n && j + h < n && text[i + h] == text[j + h]) h++;
                lcp[r] = h;
                if (h > 0) h--;
            } else {
                h = 0;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Predecessor array: pred[i] = previous SA-rank with the same doc, or -1
    // (Same structure as AlgoFHK, enables O(rb-lb) df counting.)
    // -------------------------------------------------------------------------

    private void buildPred() {
        pred = new int[n];
        int[] lastSeen = new int[numDocs + 1];
        Arrays.fill(lastSeen, -1);
        for (int i = 0; i < n; i++) {
            int d = docId[sa[i]];
            if (d >= 0 && d < numDocs) {
                pred[i]    = lastSeen[d];
                lastSeen[d] = i;
            } else {
                pred[i] = -1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Sparse table for O(1) range-minimum on lcp[]
    // -------------------------------------------------------------------------

    private void buildRMQ() {
        int maxK = 1;
        while ((1 << maxK) <= n) maxK++;

        rmqLog = new int[n + 1];
        for (int i = 2; i <= n; i++) rmqLog[i] = rmqLog[i >> 1] + 1;

        rmqTable = new int[maxK][n];
        System.arraycopy(lcp, 0, rmqTable[0], 0, n);
        for (int k = 1; k < maxK; k++) {
            int half = 1 << (k - 1);
            for (int i = 0; i + (1 << k) <= n; i++) {
                rmqTable[k][i] = Math.min(rmqTable[k-1][i], rmqTable[k-1][i + half]);
            }
        }
    }

    /** Minimum value of lcp[l..r] (inclusive), O(1). */
    private int rmq(int l, int r) {
        if (l > r)  return Integer.MAX_VALUE;
        if (l == r) return lcp[l];
        int k = rmqLog[r - l + 1];
        return Math.min(rmqTable[k][l], rmqTable[k][r - (1 << k) + 1]);
    }

    // -------------------------------------------------------------------------
    // Document-frequency counting for SA range [lb..rb]
    // -------------------------------------------------------------------------

    private int countDf(int lb, int rb) {
        int df = 0;
        for (int k = lb; k <= rb; k++) {
            if (pred[k] < lb) df++;
        }
        return df;
    }

    // -------------------------------------------------------------------------
    // Left-maximality check  (identical to AlgoFHK)
    // -------------------------------------------------------------------------

    private boolean isLeftMaximal(int lb, int rb) {
        HashMap<Integer, boolean[]> perPredDocs = null;
        for (int i = lb; i <= rb; i++) {
            int suf = sa[i];
            if (suf == 0) continue;
            if (docId[suf - 1] != docId[suf]) continue;
            int predToken = text[suf - 1];
            int doc       = docId[suf];
            if (perPredDocs == null) perPredDocs = new HashMap<>(4);
            boolean[] seen = perPredDocs.computeIfAbsent(predToken, k -> new boolean[numDocs]);
            seen[doc] = true;
        }
        if (perPredDocs == null) return true;
        for (Map.Entry<Integer, boolean[]> e : perPredDocs.entrySet()) {
            int df = 0;
            for (boolean v : e.getValue()) if (v) df++;
            if (df >= minSupportAbs) return false;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Core mining: top-down DFI traversal
    // -------------------------------------------------------------------------

    private List<AlgoBloomSpan.Phrase> mineMCFPs() {
        List<AlgoBloomSpan.Phrase> outPhrases = new ArrayList<>();
        if (n <= 1) return outPhrases;

        /*
         * Iterative top-down DFS over the virtual LCP tree.
         *
         * Each internal node corresponds to a SA range [lb, rb] whose
         * string-depth (shared prefix length) is  nodeDepth = min(lcp[lb+1..rb]).
         *
         * DFI's key step: count df(lb,rb) first; if df < minSupportAbs skip the
         * whole range — no child is ever examined.
         *
         * For each surviving node we:
         *  1. Scan [lb+1..rb] once to find split points (lcp[i] == nodeDepth),
         *     partitioning the range into children.
         *  2. Count df for each child, track maxChildDf, push frequent children.
         *  3. Emit the node if right-maximal (maxChildDf < minSup) and left-maximal
         *     and nodeDepth >= minLen.
         */
        Deque<int[]> stack = new ArrayDeque<>();
        stack.push(new int[]{0, n - 1});

        while (!stack.isEmpty()) {
            int[] frame = stack.pop();
            int lb = frame[0], rb = frame[1];
            if (lb == rb) continue;                      // leaf — no pattern to emit

            int nodeDepth = rmq(lb + 1, rb);             // string-depth of this node

            // --- DFI early pruning ---
            int df = countDf(lb, rb);
            if (df < minSupportAbs) continue;

            // --- Find children and compute maxChildDf in one linear scan ---
            // Children are delimited by positions i where lcp[i] == nodeDepth.
            // The sentinel at i = rb+1 closes the last child.
            int maxChildDf = 0;
            int childLb = lb;

            for (int i = lb + 1; i <= rb + 1; i++) {
                // curLcp is the LCP between sa[i-1] and sa[i]; sentinel = -1
                int curLcp = (i <= rb) ? lcp[i] : -1;

                if (curLcp <= nodeDepth) {               // end of current child
                    int childRb = i - 1;
                    int childDf = countDf(childLb, childRb);

                    if (childDf > maxChildDf) maxChildDf = childDf;

                    // Push only non-leaf children that are frequent enough
                    if (childDf >= minSupportAbs && childLb < childRb) {
                        stack.push(new int[]{childLb, childRb});
                    }
                    childLb = i;
                }
            }

            // --- Maximality checks ---
            boolean rightMaximal = (maxChildDf < minSupportAbs);
            if (rightMaximal && nodeDepth >= minLen && isLeftMaximal(lb, rb)) {
                int start = sa[lb];
                int[] phraseTokens = new int[nodeDepth];
                for (int p = 0; p < nodeDepth; p++) {
                    phraseTokens[p] = text[start + p];
                }

                List<AlgoBloomSpan.Occurrence> occurrences = new ArrayList<>();
                for (int k = lb; k <= rb; k++) {
                    if (pred[k] < lb) {
                        int doc    = docId[sa[k]];
                        int docPos = sa[k] - docStart[doc];
                        occurrences.add(new AlgoBloomSpan.Occurrence(doc, docPos));
                    }
                }
                outPhrases.add(new AlgoBloomSpan.Phrase(phraseTokens, occurrences, df));
            }
        }

        return outPhrases;
    }

    // -------------------------------------------------------------------------
    // SA-IS suffix array construction  (copied verbatim from AlgoFHK)
    // -------------------------------------------------------------------------

    static final class SAIS {

        private static void induceL(int[] T, int[] SA, int[] buckets,
                                    byte[] type, int n, int K) {
            bucketHeads(T, buckets, K, n);
            for (int i = 0; i < n; i++) {
                int j = SA[i] - 1;
                if (j >= 0 && type[j] == 0) {
                    int c = T[j];
                    SA[buckets[c]++] = j;
                }
            }
        }

        private static void induceS(int[] T, int[] SA, int[] buckets,
                                    byte[] type, int n, int K) {
            bucketTails(T, buckets, K, n);
            for (int i = n - 1; i >= 0; i--) {
                int j = SA[i] - 1;
                if (j >= 0 && type[j] != 0) {
                    int c = T[j];
                    SA[--buckets[c]] = j;
                }
            }
        }

        private static void getCounts(int[] T, int[] buckets, int K, int n) {
            Arrays.fill(buckets, 0, K, 0);
            for (int i = 0; i < n; i++) buckets[T[i]]++;
        }

        private static void bucketHeads(int[] T, int[] buckets, int K, int n) {
            getCounts(T, buckets, K, n);
            int sum = 0;
            for (int i = 0; i < K; i++) { int c = buckets[i]; buckets[i] = sum; sum += c; }
        }

        private static void bucketTails(int[] T, int[] buckets, int K, int n) {
            getCounts(T, buckets, K, n);
            int sum = 0;
            for (int i = 0; i < K; i++) { sum += buckets[i]; buckets[i] = sum; }
        }

        static void build(int[] T, int[] SA, int n, int K) { saIS(T, SA, n, K); }

        private static boolean isLMS(int i, byte[] type) {
            return i > 0 && type[i] == 1 && type[i - 1] == 0;
        }

        private static void saIS(int[] T, int[] SA, int n, int K) {
            byte[] type = new byte[n];
            type[n - 1] = 1;
            for (int i = n - 2; i >= 0; i--) {
                if      (T[i] < T[i + 1]) type[i] = 1;
                else if (T[i] == T[i + 1]) type[i] = type[i + 1];
                else                        type[i] = 0;
            }

            int[] buckets = new int[K];
            bucketTails(T, buckets, K, n);
            Arrays.fill(SA, -1);
            for (int i = 1; i < n; i++) {
                if (isLMS(i, type)) SA[--buckets[T[i]]] = i;
            }
            induceL(T, SA, buckets, type, n, K);
            induceS(T, SA, buckets, type, n, K);

            int n1 = 0;
            for (int i = 0; i < n; i++) {
                int p = SA[i];
                if (p > 0 && isLMS(p, type)) SA[n1++] = p;
            }
            Arrays.fill(SA, n1, n, -1);

            int name = 0, prev = -1;
            for (int i = 0; i < n1; i++) {
                int pos = SA[i];
                boolean diff = false;
                if (prev == -1) {
                    diff = true;
                } else {
                    int d = 0;
                    while (true) {
                        if (prev + d >= n || pos + d >= n ||
                                T[prev+d] != T[pos+d] || type[prev+d] != type[pos+d]) {
                            diff = true; break;
                        }
                        if (d > 0 && (isLMS(prev+d, type) || isLMS(pos+d, type))) {
                            if (isLMS(prev+d, type) != isLMS(pos+d, type)) diff = true;
                            break;
                        }
                        d++;
                    }
                }
                if (diff) { name++; prev = pos; }
                SA[n1 + (pos >>> 1)] = name - 1;
            }

            int j = n - 1;
            for (int i = n - 1; i >= n1; i--) {
                if (SA[i] >= 0) SA[j--] = SA[i];
            }

            int[] T1  = new int[n1];
            System.arraycopy(SA, n - n1, T1, 0, n1);
            int[] SA1 = new int[n1];

            if (name < n1) {
                saIS(T1, SA1, n1, name);
            } else {
                for (int i = 0; i < n1; i++) SA1[T1[i]] = i;
            }

            int k = 0;
            for (int i = 1; i < n; i++) {
                if (isLMS(i, type)) SA[n - n1 + (k++)] = i;
            }
            for (int i = 0; i < n1; i++) SA1[i] = SA[n - n1 + SA1[i]];

            bucketTails(T, buckets, K, n);
            Arrays.fill(SA, -1);
            for (int i = n1 - 1; i >= 0; i--) {
                int p = SA1[i];
                SA[--buckets[T[p]]] = p;
            }
            induceL(T, SA, buckets, type, n, K);
            induceS(T, SA, buckets, type, n, K);
        }
    }
}
