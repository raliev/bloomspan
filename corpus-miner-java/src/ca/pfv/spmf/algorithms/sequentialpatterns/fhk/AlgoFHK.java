package ca.pfv.spmf.algorithms.sequentialpatterns.fhk;

import java.io.*;
import java.util.*;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.AlgoBloomSpan;
import ca.pfv.spmf.tools.MemoryLogger;

public class AlgoFHK {

    private int minSupportAbs;
    private int minLen = 1;

    private int numDocs;
    private int tokenCount;
    private int vocabSize;
    private int[] text;
    private int n;
    private int alphabetSize;

    private int[] docId;
    private int[] docStart;
    private int[] docLen;

    private int[] sa;
    private int[] lcp;

    public void setMinL(int minL) {
        this.minLen = minL;
    }

    public List<AlgoBloomSpan.Phrase> runAlgorithm(String input, String output, double minsupRel) throws IOException {
        MemoryLogger.getInstance().reset();

        loadCorpus(input);
        MemoryLogger.getInstance().checkMemory();

        this.minSupportAbs = (int) Math.ceil(minsupRel * numDocs);
        if (this.minSupportAbs == 0) {
            this.minSupportAbs = 1;
        }

        buildSuffixArray();
        MemoryLogger.getInstance().checkMemory();
        buildLCP();
        MemoryLogger.getInstance().checkMemory();
        List<AlgoBloomSpan.Phrase> result = mineMCFPs();
        MemoryLogger.getInstance().checkMemory();
        System.out.println("MAXMEMORY=" + MemoryLogger.getInstance().getMaxMemory());
        return result;
    }

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

        numDocs = perDocTokens.size();
        vocabSize = maxId;
        tokenCount = (int) Math.min(totalTokens, Integer.MAX_VALUE);

        n = tokenCount + numDocs + 1;
        text = new int[n];
        docId = new int[n];
        docStart = new int[numDocs];
        docLen = new int[numDocs];

        int pos = 0;
        for (int d = 0; d < numDocs; d++) {
            int[] ids = perDocTokens.get(d);
            docStart[d] = pos;
            docLen[d] = ids.length;
            for (int t : ids) {
                text[pos] = t;
                docId[pos] = d;
                pos++;
            }
            text[pos] = vocabSize + 1 + d;
            docId[pos] = d;
            pos++;
        }
        text[pos] = 0;
        docId[pos] = numDocs;
        pos++;

        if (pos != n) {
            throw new IllegalStateException("Text construction length mismatch: pos=" + pos + " n=" + n);
        }

        alphabetSize = vocabSize + 1 + numDocs;
    }

    private void buildSuffixArray() {
        sa = new int[n];
        SAIS.build(text, sa, n, alphabetSize);
    }

    private void buildLCP() {
        lcp = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++)
            rank[sa[i]] = i;
        int h = 0;
        for (int i = 0; i < n; i++) {
            int r = rank[i];
            if (r > 0) {
                int j = sa[r - 1];
                while (i + h < n && j + h < n && text[i + h] == text[j + h])
                    h++;
                lcp[r] = h;
                if (h > 0)
                    h--;
            } else {
                h = 0;
            }
        }
    }

    private boolean isLeftMaximal(int lb, int rb) {
        HashMap<Integer, boolean[]> perPredDocs = null;

        for (int i = lb; i <= rb; i++) {
            int suf = sa[i];
            if (suf == 0)
                continue;
            if (docId[suf - 1] != docId[suf])
                continue;
            int pred = text[suf - 1];
            int doc = docId[suf];
            if (perPredDocs == null)
                perPredDocs = new HashMap<>(4);
            boolean[] seen = perPredDocs.computeIfAbsent(pred, k -> new boolean[numDocs]);
            seen[doc] = true;
        }

        if (perPredDocs == null)
            return true;

        for (Map.Entry<Integer, boolean[]> e : perPredDocs.entrySet()) {
            int df = 0;
            for (boolean v : e.getValue())
                if (v)
                    df++;
            if (df >= minSupportAbs)
                return false;
        }
        return true;
    }

    private List<AlgoBloomSpan.Phrase> mineMCFPs() {
        List<AlgoBloomSpan.Phrase> outPhrases = new ArrayList<>();

        int[] pred = new int[n];
        int[] lastSeenDoc = new int[numDocs + 1];
        Arrays.fill(lastSeenDoc, -1);
        for (int i = 0; i < n; i++) {
            int d = docId[sa[i]];
            if (d >= 0 && d < numDocs) {
                pred[i] = lastSeenDoc[d];
                lastSeenDoc[d] = i;
            } else {
                pred[i] = -1;
            }
        }

        int stackCap = 64;
        int[] stkLb = new int[stackCap];
        int[] stkDepth = new int[stackCap];
        int[] stkMaxChildDf = new int[stackCap];
        int top = -1;

        top++;
        stkLb[top] = 0;
        stkDepth[top] = 0;
        stkMaxChildDf[top] = 0;

        int lastLb;
        for (int i = 1; i <= n; i++) {
            int curLcp = (i == n) ? 0 : lcp[i];
            lastLb = i - 1;

            int pendingChildMaxDf = 0;

            while (stkDepth[top] > curLcp) {
                int lb = stkLb[top];
                int depth = stkDepth[top];
                int maxChildDf = Math.max(stkMaxChildDf[top], pendingChildMaxDf);
                int rb = i - 1;

                int df = 0;
                for (int k = lb; k <= rb; k++) {
                    if (pred[k] < lb)
                        df++;
                }

                if (df >= minSupportAbs) {
                    boolean rightMax = (maxChildDf < minSupportAbs);
                    if (rightMax && depth >= minLen) {
                        if (isLeftMaximal(lb, rb)) {
                            int start = sa[lb];
                            int[] phraseTokens = new int[depth];
                            for (int p = 0; p < depth; p++) {
                                phraseTokens[p] = text[start + p];
                            }
                            
                            List<AlgoBloomSpan.Occurrence> occurrences = new ArrayList<>();
                            for (int k = lb; k <= rb; k++) {
                                if (pred[k] < lb) {
                                    int doc = docId[sa[k]];
                                    int docPos = sa[k] - docStart[doc];
                                    occurrences.add(new AlgoBloomSpan.Occurrence(doc, docPos));
                                }
                            }
                            outPhrases.add(new AlgoBloomSpan.Phrase(phraseTokens, occurrences, df));
                        }
                    }
                }

                if (df > pendingChildMaxDf)
                    pendingChildMaxDf = df;

                top--;
                lastLb = lb;
            }

            if (i == n)
                break;

            if (stkDepth[top] == curLcp) {
                if (pendingChildMaxDf > stkMaxChildDf[top]) {
                    stkMaxChildDf[top] = pendingChildMaxDf;
                }
            } else {
                if (top + 1 >= stackCap) {
                    stackCap *= 2;
                    stkLb = Arrays.copyOf(stkLb, stackCap);
                    stkDepth = Arrays.copyOf(stkDepth, stackCap);
                    stkMaxChildDf = Arrays.copyOf(stkMaxChildDf, stackCap);
                }
                top++;
                stkLb[top] = lastLb;
                stkDepth[top] = curLcp;
                stkMaxChildDf[top] = pendingChildMaxDf;
            }
        }

        return outPhrases;
    }

    static final class SAIS {
        private static void induceL(int[] T, int[] SA, int[] buckets, byte[] type, int n, int K) {
            bucketHeads(T, buckets, K, n);
            for (int i = 0; i < n; i++) {
                int j = SA[i] - 1;
                if (j >= 0 && type[j] == 0) {
                    int c = T[j];
                    SA[buckets[c]++] = j;
                }
            }
        }

        private static void induceS(int[] T, int[] SA, int[] buckets, byte[] type, int n, int K) {
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
            for (int i = 0; i < n; i++)
                buckets[T[i]]++;
        }

        private static void bucketHeads(int[] T, int[] buckets, int K, int n) {
            getCounts(T, buckets, K, n);
            int sum = 0;
            for (int i = 0; i < K; i++) {
                int c = buckets[i];
                buckets[i] = sum;
                sum += c;
            }
        }

        private static void bucketTails(int[] T, int[] buckets, int K, int n) {
            getCounts(T, buckets, K, n);
            int sum = 0;
            for (int i = 0; i < K; i++) {
                sum += buckets[i];
                buckets[i] = sum;
            }
        }

        static void build(int[] T, int[] SA, int n, int K) {
            saIS(T, SA, n, K);
        }

        private static boolean isLMS(int i, byte[] type) {
            return i > 0 && type[i] == 1 && type[i - 1] == 0;
        }

        private static void saIS(int[] T, int[] SA, int n, int K) {
            byte[] type = new byte[n];
            type[n - 1] = 1;
            for (int i = n - 2; i >= 0; i--) {
                if (T[i] < T[i + 1]) {
                    type[i] = 1;
                } else if (T[i] == T[i + 1]) {
                    type[i] = type[i + 1];
                } else {
                    type[i] = 0;
                }
            }

            int[] buckets = new int[K];
            bucketTails(T, buckets, K, n);
            Arrays.fill(SA, -1);
            for (int i = 1; i < n; i++) {
                if (isLMS(i, type)) {
                    int c = T[i];
                    SA[--buckets[c]] = i;
                }
            }

            induceL(T, SA, buckets, type, n, K);
            induceS(T, SA, buckets, type, n, K);

            int n1 = 0;
            for (int i = 0; i < n; i++) {
                int p = SA[i];
                if (p > 0 && isLMS(p, type)) {
                    SA[n1++] = p;
                }
            }
            Arrays.fill(SA, n1, n, -1);

            int name = 0;
            int prev = -1;
            for (int i = 0; i < n1; i++) {
                int pos = SA[i];
                boolean diff = false;
                if (prev == -1) {
                    diff = true;
                } else {
                    int d = 0;
                    while (true) {
                        if (prev + d >= n || pos + d >= n ||
                                T[prev + d] != T[pos + d] ||
                                type[prev + d] != type[pos + d]) {
                            diff = true;
                            break;
                        }
                        if (d > 0 && (isLMS(prev + d, type) || isLMS(pos + d, type))) {
                            if (isLMS(prev + d, type) != isLMS(pos + d, type)) {
                                diff = true;
                            }
                            break;
                        }
                        d++;
                    }
                }
                if (diff) {
                    name++;
                    prev = pos;
                }
                SA[n1 + (pos >>> 1)] = name - 1;
            }

            int j = n - 1;
            for (int i = n - 1; i >= n1; i--) {
                if (SA[i] >= 0) {
                    SA[j--] = SA[i];
                }
            }

            int[] T1 = new int[n1];
            System.arraycopy(SA, n - n1, T1, 0, n1);

            int[] SA1 = new int[n1];

            if (name < n1) {
                saIS(T1, SA1, n1, name);
            } else {
                for (int i = 0; i < n1; i++)
                    SA1[T1[i]] = i;
            }

            int k = 0;
            for (int i = 1; i < n; i++) {
                if (isLMS(i, type)) {
                    SA[n - n1 + (k++)] = i;
                }
            }

            for (int i = 0; i < n1; i++) {
                SA1[i] = SA[n - n1 + SA1[i]];
            }

            bucketTails(T, buckets, K, n);
            Arrays.fill(SA, -1);
            for (int i = n1 - 1; i >= 0; i--) {
                int p = SA1[i];
                int c = T[p];
                SA[--buckets[c]] = p;
            }

            induceL(T, SA, buckets, type, n, K);
            induceS(T, SA, buckets, type, n, K);
        }
    }
}
