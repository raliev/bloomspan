package ca.pfv.spmf.algorithms.sequentialpatterns.spam;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.IntStream;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;
import ca.pfv.spmf.tools.MemoryLogger;

/**
 * Port of BloomSpan (BloomNgramMiner) from C++ to Java.
 * This algorithm finds maximal contiguous sequential patterns using a Bloom
 * Filter
 * for frequency estimation and DFS for expansion.
 */
public class AlgoBloomSpan {

    private long startTime;
    private long startTimeAfterLoadDatabase;
    private long endTime;
    private int patternCount;
    private int minsup;
    private int ngrams = 2; // Default seed size
    private int minL = 1; // Minimum length of final phrases
    private long totalProcessedNgrams = 0;

    private List<int[]> database;
    private Map<Integer, Integer> wordDF;

    public AlgoBloomSpan() {
    }

    public List<Phrase> runAlgorithm(String input, String output, double minsupRel) throws IOException {
        patternCount = 0;
        MemoryLogger.getInstance().reset();
        startTime = System.currentTimeMillis();

        // 1. Load Database
        loadDatabase(input);
        startTimeAfterLoadDatabase = System.currentTimeMillis();
        MemoryLogger.getInstance().checkMemory();

        this.minsup = (int) Math.ceil(minsupRel * database.size());
        if (this.minsup == 0)
            this.minsup = 1;

        System.out.println("[START] Beginning mining with algorithm=bloomspan, min_docs=" + this.minsup + ", ngrams="
                + this.ngrams);

        long totalWords = database.stream().mapToLong(seq -> seq.length).sum();
        int maxFilterSize = 1024 * 1024 * 512;

        int filterSize = (int) Math.min((long) maxFilterSize, Math.max(1024 * 1024, totalWords * 8));
        long filterSizeMB = Math.max(1, filterSize / (1024 * 1024));
        System.out.println("[LOG] Initializing Bloom Filter: " + filterSize + " bytes (~" + filterSizeMB + " MB)");
        byte[] filterCounters = new byte[filterSize];
        System.out.println("[LOG] Bloom Pass: Estimating n-gram frequencies...");
        estimateFrequencies(filterCounters);

        // 3. Gathering Seeds
        System.out.println("[LOG] Step 1: Gathering " + this.ngrams + "-gram seeds...");
        long s1Start = System.currentTimeMillis();
        List<SeedEntry> seeds = gatherSeeds(filterCounters);

        long totalNgrams = this.totalProcessedNgrams;
        long accepted = seeds.size();
        long rejected = totalNgrams - accepted;
        double efficiency = totalNgrams > 0 ? (100.0 * rejected / totalNgrams) : 0;
        System.out.println("[BLOOM STATS] Total n-grams: " + totalNgrams);
        System.out.println("[BLOOM STATS] Accepted:    " + accepted);
        System.out.printf("[BLOOM STATS] Rejected:    %d (%.0f%% reduction)\n", rejected, efficiency);

        filterCounters = null;

        System.out.println();
        System.out.println("[LOG] Flushing " + seeds.size() + " seeds to disk... (RAM: " + getUsedMemoryMB() + " MB)");
        System.out.println();

        // 4. Sort and Merge Seeds into initial Candidates
        System.out.println("[LOG] Step 1.5: Merging and filtering candidates...");
        SeedEntry[] seedArray = seeds.toArray(new SeedEntry[0]);
        seeds = null;
        Arrays.parallelSort(seedArray);
        List<Phrase> candidates = mergeSeeds(seedArray);
        seedArray = null;


        long s1End = System.currentTimeMillis();
        System.out.printf("[TIMER] %d-gram Seed Generation (Disk): %.5f seconds\n", this.ngrams,
                (s1End - s1Start) / 1000.0);

        System.out.println("[LOG] Step 2: Sorting " + candidates.size() + " candidates by score (support * length)...");

        // 5. Expansion with Path Compression
        System.out.println("[LOG] Step 3: Expanding with Path Compression (Jumps)...");
        long s3Start = System.currentTimeMillis();
        List<Phrase> finalPhrases = expandCandidates(candidates);
        System.out.println();
        long s3End = System.currentTimeMillis();
        System.out.printf("[TIMER] Expansion & Pruning: %.5f seconds\n", (s3End - s3Start) / 1000.0);

        MemoryLogger.getInstance().checkMemory();

        endTime = System.currentTimeMillis();

        long count6plus = finalPhrases.stream().filter(p -> p.tokens.length >= 6).count();
        long totalSeedsGenerated = candidates.size();

        System.out.println("\n========== MINING STATISTICS ==========");
        System.out.println("Candidates after merge:       " + totalSeedsGenerated);
        System.out.println("Total phrases mined:          " + finalPhrases.size());
        System.out.println("Long phrases (6+ words):      " + count6plus);
        System.out.println("=======================================\n");
        System.out.println("MAXMEMORY=" + MemoryLogger.getInstance().getMaxMemory());

        System.out.printf("[TIMER] Total Process with Loading: %.5f seconds\n", (endTime - startTime) / 1000.0);
        System.out.printf("[TIMER] Total Mining Process: %.5f seconds\n",
                (endTime - startTimeAfterLoadDatabase) / 1000.0);

        return finalPhrases;
    }

    private long getUsedMemoryMB() {
        Runtime runtime = Runtime.getRuntime();
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
    }

    private void loadDatabase(String input) throws IOException {
        database = new ArrayList<>();
        wordDF = new HashMap<>();

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(input))));
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("%") || line.startsWith("@"))
                continue;

            String[] tokens = line.split(" ");
            List<Integer> sequence = new ArrayList<>();
            Set<Integer> uniqueInDoc = new HashSet<>();
            for (String t : tokens) {
                int item = Integer.parseInt(t);
                if (item >= 0) {
                    sequence.add(item);
                    uniqueInDoc.add(item);
                }
            }
            database.add(sequence.stream().mapToInt(i -> i).toArray());
            for (int item : uniqueInDoc) {
                wordDF.put(item, wordDF.getOrDefault(item, 0) + 1);
            }
        }
        reader.close();
    }

    public void setMinL(int minL) {
        this.minL = minL;
    }

    public void setNgrams(int ngrams) {
        this.ngrams = ngrams;
    }

    private void estimateFrequencies(byte[] filter) {
        IntStream.range(0, database.size()).parallel().forEach(d -> {
            int[] seq = database.get(d);
            if (seq.length < ngrams)
                return;
            for (int p = 0; p <= seq.length - ngrams; p++) {
                int idx = (int) (Math.abs(hashTokens(seq, p, ngrams)) % filter.length);
                int col = filter[idx] & 0xFF;
                if (col < 255) {
                    filter[idx] = (byte) (col + 1);
                }
            }
        });
    }

    private long hashTokens(int[] seq, int start, int n) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < n; i++) {
            h ^= seq[start + i];
            h *= 0x100000001b3L;
        }
        return h;
    }

    private List<SeedEntry> gatherSeeds(byte[] filter) {
        AtomicLong totalProcessed = new AtomicLong(0);
        AtomicLong seedsPassed = new AtomicLong(0);
        AtomicInteger docsProcessed = new AtomicInteger(0);

        List<SeedEntry> seeds = IntStream.range(0, database.size()).parallel().mapToObj(d -> {
            List<SeedEntry> buffer = new ArrayList<>();
            int[] seq = database.get(d);
            if (seq.length < ngrams) {
                int dp = docsProcessed.incrementAndGet();
                if (dp % 500 == 0 || dp == database.size()) {
                    System.out.print("\r[LOG] Scanning: " + dp + "/" + database.size() + " | Seeds Found: "
                            + seedsPassed.get() + " ");
                }
                return buffer;
            }

            totalProcessed.addAndGet(seq.length - ngrams + 1);
            for (int p = 0; p <= seq.length - ngrams; p++) {
                long h = hashTokens(seq, p, ngrams);
                int count = filter[(int) (Math.abs(h) % filter.length)] & 0xFF;
                if (count >= Math.min(minsup, 255)) {
                    boolean dfOk = true;
                    for (int i = 0; i < ngrams; i++) {
                        if (wordDF.getOrDefault(seq[p + i], 0) < minsup) {
                            dfOk = false;
                            break;
                        }
                    }
                    if (dfOk) {
                        buffer.add(new SeedEntry(d, p, seq, ngrams));
                        seedsPassed.incrementAndGet();
                    }
                }
            }
            int dp = docsProcessed.incrementAndGet();
            if (dp % 500 == 0 || dp == database.size()) {
                System.out.print("\r[LOG] Scanning: " + dp + "/" + database.size() + " | Seeds Found: "
                        + seedsPassed.get() + " ");
            }
            return buffer;
        }).flatMap(List::stream).collect(java.util.stream.Collectors.toList());
        System.out.println();
        this.totalProcessedNgrams = totalProcessed.get();
        return seeds;
    }

    private List<Phrase> mergeSeeds(SeedEntry[] seeds) {
        List<Phrase> candidates = new ArrayList<>();
        int i = 0;
        while (i < seeds.length) {
            SeedEntry rep = seeds[i];
            List<Occurrence> occs = new ArrayList<>();
            Set<Integer> uniqueDocs = new HashSet<>();
            while (i < seeds.length && seeds[i].sameTokens(rep)) {
                occs.add(new Occurrence(seeds[i].docId, seeds[i].pos));
                uniqueDocs.add(seeds[i].docId);
                i++;
            }
            if (uniqueDocs.size() >= minsup) {
                candidates.add(new Phrase(rep.getTokens(ngrams), occs, uniqueDocs.size()));
            }
        }
        return candidates;
    }

    private List<Phrase> expandCandidates(List<Phrase> candidates) {
        // 1. Deterministic Sort: Score (desc), then Length (desc), then Lexicographical
        // This ensures super-patterns like "b c d e" (Score 28) are processed before "c
        // d e" (Score 24)
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

        // 2. Global processed tracking (similar to C++ processed matrix)
        boolean[][] processed = new boolean[database.size()][];
        for (int i = 0; i < database.size(); i++)
            processed[i] = new boolean[database.get(i).length];

        List<Phrase> finalPhrases = new ArrayList<>();

        int cIdx = 0;
        for (Phrase cand : candidates) {
            if (cIdx % 100 == 0 || cIdx == candidates.size() - 1) {
                System.out.print("\r[LOG] Expanding: " + (cIdx + 1) + "/" + candidates.size()
                        + " | Phrases found: " + finalPhrases.size() + "          ");
            }
            cIdx++;

            // Pruning: skip seed if all its tokens in all occurrences are already covered
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

            Stack<Phrase> stack = new Stack<>();
            stack.push(cand);
            MemoryLogger.getInstance().checkMemory();
            while (!stack.isEmpty()) {
                Phrase current = stack.pop();
                Map<Integer, List<Occurrence>> extensions = new HashMap<>();

                for (Occurrence o : current.occs) {
                    int[] doc = database.get(o.docId);
                    int nextPos = o.pos + current.tokens.length;
                    if (nextPos < doc.length) {
                        extensions.computeIfAbsent(doc[nextPos], k -> new ArrayList<>())
                                .add(new Occurrence(o.docId, o.pos));
                        MemoryLogger.getInstance().checkMemory();
                    }
                }

                boolean expanded = false;
                for (Map.Entry<Integer, List<Occurrence>> entry : extensions.entrySet()) {
                    Set<Integer> uDocs = new HashSet<>();
                    for (Occurrence o : entry.getValue())
                        uDocs.add(o.docId);

                    if (uDocs.size() >= minsup) {
                        int[] nextTokens = Arrays.copyOf(current.tokens, current.tokens.length + 1);
                        nextTokens[current.tokens.length] = entry.getKey();
                        stack.push(new Phrase(nextTokens, entry.getValue(), uDocs.size()));
                        expanded = true;
                    }
                    MemoryLogger.getInstance().checkMemory();
                }

                if (!expanded) {
                    // 3. Bi-directional Maximality Check
                    if (isMaximal(current, finalPhrases)) {
                        // Mark tokens in the global processed matrix
                        for (Occurrence o : current.occs) {
                            for (int k = 0; k < current.tokens.length; k++) {
                                if (o.pos + k < processed[o.docId].length)
                                    processed[o.docId][o.pos + k] = true;
                            }
                        }
                        finalPhrases.add(current);
                    }
                }
                MemoryLogger.getInstance().checkMemory();
            }
        }

        patternCount = finalPhrases.size();
        return finalPhrases;
    }

    private boolean isMaximal(Phrase current, List<Phrase> existing) {
        if (current.tokens.length < minL)
            return false;

        // A. Backward-extension check (Local)
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

        // B. Global Bi-directional Check (Matching C++ search logic)
        Iterator<Phrase> it = existing.iterator();
        while (it.hasNext()) {
            Phrase p = it.next();
            // If current is a sub-phrase of something already found, current is not maximal
            if (p.tokens.length >= current.tokens.length) {
                if (isSubArray(p.tokens, current.tokens))
                    return false;
            }
            // If something found earlier is a sub-phrase of current, remove it
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

    public void printStatistics() {
        System.out.println("============= BloomSpan STATISTICS =============");
        System.out.println(" Total time: " + (endTime - startTime) + " ms");
        System.out.println(" Frequent sequences count: " + patternCount);
        System.out.println(" Max memory: " + MemoryLogger.getInstance().getMaxMemory() + " MB");
        System.out.println("================================================");
    }

    // Helper classes
    public static class SeedEntry implements Comparable<SeedEntry> {
        public int docId, pos;
        public long packedTokens; // For ngrams=2
        public int[] tokens; // For ngrams > 2

        public SeedEntry(int d, int p, int[] seq, int ngrams) {
            this.docId = d;
            this.pos = p;
            if (ngrams == 2) {
                this.packedTokens = ((long) seq[p] << 32) | (seq[p + 1] & 0xFFFFFFFFL);
            } else {
                this.tokens = Arrays.copyOfRange(seq, p, p + ngrams);
            }
        }

        public int[] getTokens(int n) {
            if (tokens != null)
                return tokens;
            return new int[] { (int) (packedTokens >> 32), (int) (packedTokens) };
        }

        public boolean sameTokens(SeedEntry o) {
            if (this.tokens == null) {
                return this.packedTokens == o.packedTokens;
            }
            return Arrays.equals(this.tokens, o.tokens);
        }

        @Override
        public int compareTo(SeedEntry o) {
            if (this.tokens == null) {
                if (this.packedTokens != o.packedTokens) {
                    return Long.compare(this.packedTokens, o.packedTokens);
                }
            } else {
                for (int i = 0; i < tokens.length; i++) {
                    if (this.tokens[i] != o.tokens[i])
                        return Integer.compare(this.tokens[i], o.tokens[i]);
                }
            }
            if (this.docId != o.docId)
                return Integer.compare(this.docId, o.docId);
            return Integer.compare(this.pos, o.pos);
        }
    }

    public static class Occurrence {
        public int docId, pos;

        public Occurrence(int d, int p) {
            this.docId = d;
            this.pos = p;
        }
    }

    public static class Phrase {
        public int[] tokens;
        public List<Occurrence> occs;
        public int support;

        public Phrase(int[] t, List<Occurrence> o, int s) {
            this.tokens = t;
            this.occs = o;
            this.support = s;
        }
    }
}