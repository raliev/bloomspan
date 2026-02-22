package ca.pfv.spmf.algorithms.sequentialpatterns.BIDE_and_prefixspan;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import ca.pfv.spmf.algorithms.sequentialpatterns.spam.Bitmap;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.PatternVMSP;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.PrefixVMSP;
import ca.pfv.spmf.patterns.itemset_list_integers_without_support.Itemset;
import ca.pfv.spmf.tools.MemoryLogger;

/**
 * BIDE+ (Contiguous Maximal Mode)
 * 
 * A strict adaptation of the classical BIDE+ closed sequential pattern miner
 * constrained specifically to contiguous sequence expansions.
 * It features contiguous pseudo-projection, backscan-pruning, and
 * forward/backward
 * frequent checks to ensure MAXIMAL sequences rather than CLOSED sequences.
 */
public class AlgoBIDEPlusContiguousMaximal {

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

    public AlgoBIDEPlusContiguousMaximal() {
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

        maxPatterns = new ArrayList<>();
        bideContiguous();
        MemoryLogger.getInstance().checkMemory();

        endTime = System.currentTimeMillis();
        System.out.println("MAXMEMORY=" + MemoryLogger.getInstance().getMaxMemory());
        return maxPatterns;
    }

    public void setMinL(int minL) {
        this.minL = minL;
    }

    private void loadDatabase(String input) throws IOException {
        database = new ArrayList<>();
        sequencesSize = new ArrayList<>();
        int bitIndex = 0;

        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(input))));
        String line;

        while ((line = reader.readLine()) != null) {
            if (line.isEmpty() || line.startsWith("#") || line.startsWith("%") || line.startsWith("@"))
                continue;

            sequencesSize.add(bitIndex);

            String[] tokens = line.split(" ");
            List<Integer> sequence = new ArrayList<>();
            for (String t : tokens) {
                int item = Integer.parseInt(t);
                if (item >= 0) {
                    sequence.add(item);
                } else if (item == -1) {
                    bitIndex++;
                }
            }
            database.add(sequence.stream().mapToInt(i -> i).toArray());
        }
        lastBitIndex = bitIndex - 1;
        reader.close();
    }

    private void bideContiguous() {
        Map<Integer, List<Occurrence>> itemOccurrences = new HashMap<>();

        for (int i = 0; i < database.size(); i++) {
            int[] doc = database.get(i);
            for (int pos = 0; pos < doc.length; pos++) {
                int item = doc[pos];
                itemOccurrences.computeIfAbsent(item, k -> new ArrayList<>()).add(new Occurrence(i, pos));
            }
        }

        // Process each frequent 1-item as a seed
        for (Map.Entry<Integer, List<Occurrence>> entry : itemOccurrences.entrySet()) {
            if (getUniqueDocs(entry.getValue()) >= minsup) {
                int[] prefix = new int[] { entry.getKey() };
                expand(prefix, entry.getValue());
            }
            MemoryLogger.getInstance().checkMemory();
        }
    }

    private void expand(int[] prefix, List<Occurrence> occs) {
        // 1. BackScan Pruning
        // If an item always immediately precedes this pattern's occurrences, this
        // pattern is not closed contiguous.
        if (hasBackwardExtension(occs)) {
            return;
        }

        // 2. Find frequent contiguous extensions
        Map<Integer, List<Occurrence>> nextOccs = new HashMap<>();
        for (Occurrence o : occs) {
            int[] doc = database.get(o.docId);
            int nextPos = o.pos + prefix.length;
            if (nextPos < doc.length) {
                nextOccs.computeIfAbsent(doc[nextPos], k -> new ArrayList<>()).add(new Occurrence(o.docId, o.pos));
            }
        }

        boolean hasFrequentForwardExtension = false;

        for (Map.Entry<Integer, List<Occurrence>> entry : nextOccs.entrySet()) {
            int childSupport = getUniqueDocs(entry.getValue());

            if (childSupport >= minsup) {
                hasFrequentForwardExtension = true;
                int[] newPrefix = Arrays.copyOf(prefix, prefix.length + 1);
                newPrefix[prefix.length] = entry.getKey();
                expand(newPrefix, entry.getValue());
            }
            MemoryLogger.getInstance().checkMemory();
        }

        // 3. Maximality Verification
        // If there's no frequent forward extension, and no frequent backward extension,
        // it's MAXIMAL.
        if (!hasFrequentForwardExtension) {
            if (!hasFrequentBackwardExtension(occs)) {
                if (prefix.length >= minL) {
                    saveToMaxPatterns(prefix, occs);
                }
            }
        }
    }

    private boolean hasBackwardExtension(List<Occurrence> occs) {
        if (occs.isEmpty())
            return false;
        int firstDoc = occs.get(0).docId;
        int firstPos = occs.get(0).pos;
        if (firstPos == 0)
            return false;

        int prevItem = database.get(firstDoc)[firstPos - 1];

        for (int i = 1; i < occs.size(); i++) {
            Occurrence o = occs.get(i);
            if (o.pos == 0)
                return false;
            if (database.get(o.docId)[o.pos - 1] != prevItem)
                return false;
        }
        return true;
    }

    private boolean hasFrequentBackwardExtension(List<Occurrence> occs) {
        Map<Integer, Set<Integer>> prevItemDocs = new HashMap<>(); // item -> docIds
        for (Occurrence o : occs) {
            if (o.pos > 0) {
                int prevItem = database.get(o.docId)[o.pos - 1];
                prevItemDocs.computeIfAbsent(prevItem, k -> new HashSet<>()).add(o.docId);
            }
        }
        for (Set<Integer> docs : prevItemDocs.values()) {
            if (docs.size() >= minsup) {
                return true;
            }
        }
        return false;
    }

    private int getUniqueDocs(List<Occurrence> occs) {
        Set<Integer> unique = new HashSet<>();
        for (Occurrence o : occs)
            unique.add(o.docId);
        return unique.size();
    }

    private void saveToMaxPatterns(int[] pTokens, List<Occurrence> occs) {
        int len = pTokens.length;
        while (maxPatterns.size() <= len)
            maxPatterns.add(new TreeSet<>());

        PrefixVMSP prefix = new PrefixVMSP();
        int sumEven = 0;
        int sumOdd = 0;
        for (int token : pTokens) {
            prefix.addItemset(new Itemset(token));
            if (token % 2 == 0)
                sumEven += token;
            else
                sumOdd += token;
        }

        try {
            java.lang.reflect.Field fEven = PrefixVMSP.class.getDeclaredField("sumOfEvenItems");
            fEven.setAccessible(true);
            fEven.set(prefix, sumEven);

            java.lang.reflect.Field fOdd = PrefixVMSP.class.getDeclaredField("sumOfOddItems");
            fOdd.setAccessible(true);
            fOdd.set(prefix, sumOdd);
        } catch (Exception ignored) {
        }

        Bitmap patternBitmap = null;
        try {
            java.lang.reflect.Constructor<Bitmap> constructor = Bitmap.class.getDeclaredConstructor(int.class);
            constructor.setAccessible(true);
            patternBitmap = constructor.newInstance(lastBitIndex);
        } catch (Exception ignored) {
        }

        if (patternBitmap != null) {
            for (Occurrence o : occs) {
                patternBitmap.registerBit(o.docId, o.pos, sequencesSize);
            }
        }

        int support = getUniqueDocs(occs);
        PatternVMSP pattern = new PatternVMSP(prefix, support);

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
        System.out.println("====== BIDE+ (Contiguous Maximal) STATISTICS ======");
        System.out.println(" Total time: " + (endTime - startTime) + " ms");
        System.out.println(" Mining time: " + (endTime - startTimeAfterLoadDatabase) + " ms");
        System.out.println(" Maximal sequences count: " + patternCount);
        System.out.println(" Max memory: " + MemoryLogger.getInstance().getMaxMemory() + " MB");
        System.out.println("===================================================");
    }

    private static class Occurrence {
        int docId, pos;

        Occurrence(int d, int p) {
            this.docId = d;
            this.pos = p;
        }
    }
}
