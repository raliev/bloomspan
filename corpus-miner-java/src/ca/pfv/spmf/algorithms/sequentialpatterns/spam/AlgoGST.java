package ca.pfv.spmf.algorithms.sequentialpatterns.spam;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import ca.pfv.spmf.patterns.itemset_list_integers_without_support.Itemset;
import ca.pfv.spmf.tools.MemoryLogger;

/**
 * Generalized Suffix Tree (Array) approach to Maximal Contiguous Sequential
 * Pattern Mining.
 * Provides a theoretical O(N log N) / O(N) baseline demonstrating the memory
 * bottleneck of
 * classical text mining structures when applied to large datasets.
 */
public class AlgoGST {

    private long startTime;
    private long startTimeAfterLoadDatabase;
    private long endTime;
    private int patternCount;
    private int minsup;
    private int minL = 1;

    private List<TreeSet<PatternVMSP>> maxPatterns;
    private List<int[]> database;

    private List<Integer> sequencesSize;
    private int lastBitIndex;

    private int[] T;
    private int[] doc;
    private int[] posInDoc;
    private Integer[] sa;
    private int[] lcp;

    public AlgoGST() {
    }

    public List<TreeSet<PatternVMSP>> runAlgorithm(String input, String output, double minsupRel) throws IOException {
        patternCount = 0;
        MemoryLogger.getInstance().reset();
        startTime = System.currentTimeMillis();

        loadDatabase(input);
        startTimeAfterLoadDatabase = System.currentTimeMillis();
        MemoryLogger.getInstance().checkMemory();

        this.minsup = (int) Math.ceil(minsupRel * database.size());
        if (this.minsup == 0)
            this.minsup = 1;

        buildSuffixArray();
        MemoryLogger.getInstance().checkMemory();

        buildLCP();
        MemoryLogger.getInstance().checkMemory();

        List<Phrase> candidates = extractCandidates();
        MemoryLogger.getInstance().checkMemory();

        filterMaximal(candidates);
        MemoryLogger.getInstance().checkMemory();

        endTime = System.currentTimeMillis();
        System.out.println("MAXMEMORY=" + MemoryLogger.getInstance().getMaxMemory());
        return maxPatterns;
    }

    private void loadDatabase(String input) throws IOException {
        database = new ArrayList<>();
        sequencesSize = new ArrayList<>();
        int bitIndex = 0;

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(input))));
        String line;

        List<Integer> tList = new ArrayList<>();
        List<Integer> docList = new ArrayList<>();
        List<Integer> posList = new ArrayList<>();

        int delimiter = -1;
        int currentDoc = 0;

        while ((line = reader.readLine()) != null) {
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("%") || line.startsWith("@"))
                continue;

            sequencesSize.add(bitIndex);

            String[] tokens = line.split(" ");
            List<Integer> sequence = new ArrayList<>();
            int pos = 0;
            for (String t : tokens) {
                int item = Integer.parseInt(t);
                if (item >= 0) {
                    sequence.add(item);
                    tList.add(item);
                    docList.add(currentDoc);
                    posList.add(pos++);
                } else if (item == -1) {
                    bitIndex++;
                }
            }
            database.add(sequence.stream().mapToInt(i -> i).toArray());

            // delimiter
            tList.add(delimiter--);
            docList.add(currentDoc);
            posList.add(-1);

            currentDoc++;
        }
        lastBitIndex = bitIndex - 1;
        reader.close();

        T = new int[tList.size()];
        doc = new int[docList.size()];
        posInDoc = new int[posList.size()];
        for (int i = 0; i < T.length; i++) {
            T[i] = tList.get(i);
            doc[i] = docList.get(i);
            posInDoc[i] = posList.get(i);
        }
    }

    private void buildSuffixArray() {
        sa = new Integer[T.length];
        for (int i = 0; i < T.length; i++)
            sa[i] = i;

        Arrays.sort(sa, (a, b) -> {
            int maxLen = Math.min(T.length - a, T.length - b);
            for (int i = 0; i < maxLen; i++) {
                if (T[a + i] != T[b + i])
                    return Integer.compare(T[a + i], T[b + i]);
            }
            return Integer.compare(T.length - a, T.length - b);
        });
    }

    private void buildLCP() {
        int[] rank = new int[T.length];
        for (int i = 0; i < T.length; i++)
            rank[sa[i]] = i;
        lcp = new int[T.length];
        int h = 0;
        for (int i = 0; i < T.length; i++) {
            if (rank[i] > 0) {
                int j = sa[rank[i] - 1];
                while (i + h < T.length && j + h < T.length && T[i + h] == T[j + h] && T[i + h] >= 0) {
                    h++;
                }
                lcp[rank[i]] = h;
                if (h > 0)
                    h--;
            }
            if (i % 1000 == 0)
                MemoryLogger.getInstance().checkMemory();
        }
    }

    private List<Phrase> extractCandidates() {
        List<Phrase> candidates = new ArrayList<>();
        Stack<Interval> stack = new Stack<>();
        stack.push(new Interval(0, 0, 0));

        for (int i = 1; i <= T.length; i++) {
            int l = (i < T.length) ? lcp[i] : 0;
            int start = i - 1;
            while (!stack.isEmpty() && stack.peek().length > l) {
                Interval top = stack.pop();
                top.end = i - 1;

                if (top.length >= minL && top.length >= 2) { // also apply minL filter
                    Phrase candidate = extractPhrase(top);
                    if (candidate != null && candidate.support >= minsup) {
                        candidates.add(candidate);
                    }
                }
                start = top.start;
            }
            if (stack.isEmpty() || stack.peek().length < l) {
                stack.push(new Interval(l, start, -1));
            }
            if (i % 1000 == 0)
                MemoryLogger.getInstance().checkMemory();
        }
        return candidates;
    }

    public void setMinL(int minL) {
        this.minL = minL;
    }

    private Phrase extractPhrase(Interval in) {
        Set<Integer> uDocs = new HashSet<>();
        List<Occurrence> occs = new ArrayList<>();

        int firstValidSuffix = -1;
        for (int k = in.start; k <= in.end; k++) {
            int s = sa[k];
            if (posInDoc[s] != -1) {
                if (firstValidSuffix == -1)
                    firstValidSuffix = s;
                uDocs.add(doc[s]);
                occs.add(new Occurrence(doc[s], posInDoc[s]));
            }
        }

        if (firstValidSuffix == -1)
            return null;
        int[] tokens = Arrays.copyOfRange(T, firstValidSuffix, firstValidSuffix + in.length);

        // Sort occurrences so they are deterministic like BloomSpan
        occs.sort((a, b) -> {
            if (a.docId != b.docId)
                return Integer.compare(a.docId, b.docId);
            return Integer.compare(a.pos, b.pos);
        });

        return new Phrase(tokens, occs, uDocs.size());
    }

    private void filterMaximal(List<Phrase> candidates) {
        maxPatterns = new ArrayList<>();

        candidates.sort((a, b) -> {
            int scoreA = a.support * a.tokens.length;
            int scoreB = b.support * b.tokens.length;
            if (scoreA != scoreB)
                return Integer.compare(scoreB, scoreA);
            if (a.tokens.length != b.tokens.length)
                return Integer.compare(b.tokens.length, a.tokens.length);
            for (int i = 0; i < Math.min(a.tokens.length, b.tokens.length); i++) {
                if (a.tokens[i] != b.tokens[i])
                    return Integer.compare(a.tokens[i], b.tokens[i]);
            }
            return Integer.compare(a.tokens.length, b.tokens.length);
        });

        boolean[][] processed = new boolean[database.size()][];
        for (int i = 0; i < database.size(); i++)
            processed[i] = new boolean[database.get(i).length];

        List<Phrase> finalPhrases = new ArrayList<>();

        for (Phrase cand : candidates) {
            boolean allProcessed = true;
            for (Occurrence o : cand.occs) {
                for (int i = 0; i < cand.tokens.length; i++) {
                    if (!processed[o.docId][o.pos + i]) {
                        allProcessed = false;
                        break;
                    }
                }
                if (!allProcessed)
                    break;
            }
            if (allProcessed)
                continue;

            if (isMaximal(cand, finalPhrases)) {
                for (Occurrence o : cand.occs) {
                    for (int k = 0; k < cand.tokens.length; k++) {
                        if (o.pos + k < processed[o.docId].length)
                            processed[o.docId][o.pos + k] = true;
                    }
                }
                finalPhrases.add(cand);
            }
            MemoryLogger.getInstance().checkMemory();
        }

        for (Phrase p : finalPhrases) {
            saveToMaxPatterns(p, sequencesSize, lastBitIndex);
            MemoryLogger.getInstance().checkMemory();
        }
    }

    private boolean isMaximal(Phrase current, List<Phrase> existing) {
        if (current.tokens.length < minL)
            return false;

        if (!current.occs.isEmpty()) {
            int firstDoc = current.occs.get(0).docId;
            int firstPos = current.occs.get(0).pos;
            if (firstPos > 0) {
                int commonPrev = database.get(firstDoc)[firstPos - 1];
                boolean allMatch = true;
                for (Occurrence o : current.occs) {
                    if (o.pos == 0 || database.get(o.docId)[o.pos - 1] != commonPrev) {
                        allMatch = false;
                        break;
                    }
                }
                if (allMatch)
                    return false;
            }
        }

        Iterator<Phrase> it = existing.iterator();
        while (it.hasNext()) {
            Phrase p = it.next();
            if (p.tokens.length >= current.tokens.length) {
                if (isSubArray(p.tokens, current.tokens))
                    return false;
            }
            if (current.tokens.length > p.tokens.length) {
                if (isSubArray(current.tokens, p.tokens)) {
                    it.remove();
                    patternCount--;
                }
            }
        }
        return true;
    }

    private boolean isSubArray(int[] larger, int[] smaller) {
        if (smaller.length > larger.length)
            return false;
        for (int i = 0; i <= larger.length - smaller.length; i++) {
            boolean match = true;
            for (int j = 0; j < smaller.length; j++) {
                if (larger[i + j] != smaller[j]) {
                    match = false;
                    break;
                }
            }
            if (match)
                return true;
        }
        return false;
    }

    private void saveToMaxPatterns(Phrase p, List<Integer> sequencesSize, int lastBitIndex) {
        int len = p.tokens.length;
        while (maxPatterns.size() <= len)
            maxPatterns.add(new TreeSet<>());

        PrefixVMSP prefix = new PrefixVMSP();
        int sumEven = 0;
        int sumOdd = 0;
        for (int token : p.tokens) {
            prefix.addItemset(new Itemset(token));
            if (token % 2 == 0)
                sumEven += token;
            else
                sumOdd += token;
        }
        prefix.sumOfEvenItems = sumEven;
        prefix.sumOfOddItems = sumOdd;

        Bitmap patternBitmap = new Bitmap(lastBitIndex);
        for (Occurrence o : p.occs) {
            patternBitmap.registerBit(o.docId, o.pos, sequencesSize);
        }

        PatternVMSP pattern = new PatternVMSP(prefix, p.support);
        try {
            java.lang.reflect.Field field = PatternVMSP.class.getDeclaredField("bitmap");
            field.setAccessible(true);
            field.set(pattern, patternBitmap);
        } catch (Exception ignored) {
        }

        maxPatterns.get(len).add(pattern);
        patternCount++;
    }

    public void printStatistics() {
        System.out.println("============= GST STATISTICS =============");
        System.out.println(" Total time: " + (endTime - startTime) + " ms");
        System.out.println(" Mining time: " + (endTime - startTimeAfterLoadDatabase) + " ms");
        System.out.println(" Frequent sequences count: " + patternCount);
        System.out.println(" Max memory: " + MemoryLogger.getInstance().getMaxMemory() + " MB");
        System.out.println("==========================================");
    }

    private static class Interval {
        int length, start, end;

        Interval(int l, int s, int e) {
            length = l;
            start = s;
            end = e;
        }
    }

    private static class Occurrence {
        int docId, pos;

        Occurrence(int d, int p) {
            this.docId = d;
            this.pos = p;
        }
    }

    private static class Phrase {
        int[] tokens;
        List<Occurrence> occs;
        int support;

        Phrase(int[] t, List<Occurrence> o, int s) {
            this.tokens = t;
            this.occs = o;
            this.support = s;
        }
    }
}
