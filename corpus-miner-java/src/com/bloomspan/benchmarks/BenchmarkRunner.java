package com.bloomspan.benchmarks;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.lang.reflect.Field;
import ca.pfv.spmf.algorithms.sequentialpatterns.BIDE_and_prefixspan.AlgoBIDEPlusContiguousMaximal;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.AlgoVMSP;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.PatternVMSP;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.Bitmap;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.AlgoBloomSpan;
import ca.pfv.spmf.algorithms.sequentialpatterns.spam.AlgoGST;
import ca.pfv.spmf.algorithms.sequentialpatterns.BIDE_and_prefixspan.AlgoBIDEPlusContiguous;
import ca.pfv.spmf.algorithms.sequentialpatterns.fhk.AlgoFHK;
import ca.pfv.spmf.algorithms.sequentialpatterns.dfi.AlgoDFI;

public class BenchmarkRunner {
    private static final Map<String, Integer> wordToIdMap = new HashMap<>();
    private static final Map<Integer, String> idToWordMap = new HashMap<>();
    private static final Map<Integer, String> sidToFileMap = new HashMap<>();
    private static final List<Integer> sequencesSize = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.out.println(
                    "Usage: java BenchmarkRunner <algo> <folder> <minSupportAbs> <minLen> <outputCsv> [maxDocs]");
            return;
        }

        String algo = args[0].toLowerCase();
        String folderPath = args[1];
        int minSupportAbs = Integer.parseInt(args[2]);
        int minLen = Integer.parseInt(args[3]);
        String outputCsv = args[4];
        int maxDocs = args.length > 5 ? Integer.parseInt(args[5]) : -1;

        // 1. Prepare Directory and Relative Support
        File dir = new File(folderPath);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".txt"));
        int availableDocs = (files != null) ? files.length : 0;

        if (availableDocs == 0) {
            System.out.println("No .txt files found in " + folderPath);
            return;
        }

        int numDocs = availableDocs;
        if (maxDocs > 0 && maxDocs < availableDocs) {
            numDocs = maxDocs;
        }

        double minSupportRel = (double) minSupportAbs / numDocs;
        System.out.println("numDocs=" + numDocs);
        System.out.println("minSupportAbs=" + minSupportAbs);
        System.out.println("minSupportRel=" + minSupportRel);
        System.out.println("minLen=" + minLen);
        System.out.println("algo=" + algo);

        String convertedFile = "converted_temp.txt";
        processTextDirectory(folderPath, convertedFile, maxDocs);

        // 2. Execute Algorithm
        long startTime = System.currentTimeMillis();
        List<TreeSet<PatternVMSP>> spmfResults = null;
        List<AlgoBloomSpan.Phrase> phraseResults = null;

        // We use a temporary file for SPMF's internal output to avoid
        // NullPointerException
        String spmfInternalOutput = "spmf_internal_out.txt";

        if (algo.equals("vmsp")) {
            AlgoVMSP algoInstance = new AlgoVMSP();
            algoInstance.setMaxGap(1); // Contiguous constraint
            algoInstance.showSequenceIdentifiersInOutput(true);
            spmfResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else if (algo.equals("bloomspan")) {
            AlgoBloomSpan algoInstance = new AlgoBloomSpan();
            algoInstance.setMinL(minLen);
            algoInstance.setNgrams(minLen);
            phraseResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else if (algo.equals("gst")) {
            AlgoGST algoInstance = new AlgoGST();
            algoInstance.setMinL(minLen);
            spmfResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else if (algo.equals("bidecontiguous")) {
            AlgoBIDEPlusContiguous algoInstance = new AlgoBIDEPlusContiguous();
            algoInstance.setMinL(minLen);
            spmfResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else if (algo.equals("bidecontiguousmaximal")) {
            AlgoBIDEPlusContiguousMaximal algoInstance = new AlgoBIDEPlusContiguousMaximal();
            algoInstance.setMinL(minLen);
            spmfResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else if (algo.equals("fhk")) {
            AlgoFHK algoInstance = new AlgoFHK();
            algoInstance.setMinL(minLen);
            phraseResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else if (algo.equals("dfi")) {
            AlgoDFI algoInstance = new AlgoDFI();
            algoInstance.setMinL(minLen);
            phraseResults = algoInstance.runAlgorithm(convertedFile, spmfInternalOutput, minSupportRel);
        } else {
            System.out.println("Unknown algorithm: " + algo);
            return;
        }
        long endTime = System.currentTimeMillis();

        // 3. Export to your custom CSV
        System.out.println("--- STATISTICS ---");
        System.out.println("Time_ms: " + (endTime - startTime));
        if (algo.equals("bloomspan") || algo.equals("fhk") || algo.equals("dfi")) {
            saveBloomSpanResultsToCSV(phraseResults, outputCsv);
            int count = (phraseResults != null) ? phraseResults.size() : 0;
            System.out.println("Found_Phrases: " + count);
        } else {
            saveResultsToCSV(spmfResults, outputCsv);
            System.out.println("Found_Phrases: " + countPatterns(spmfResults));
        }

        // Clean up temporary files
        new File(convertedFile).delete();
        new File(spmfInternalOutput).delete();
    }

    private static void saveBloomSpanResultsToCSV(List<AlgoBloomSpan.Phrase> results, String csvPath) throws Exception {
        if (results == null)
            return;

        try (PrintWriter pw = new PrintWriter(new FileWriter(csvPath))) {
            pw.println("phrase,freq,length,example_files");
            for (AlgoBloomSpan.Phrase pattern : results) {
                String phrase = Arrays.stream(pattern.tokens)
                        .mapToObj(idToWordMap::get)
                        .collect(Collectors.joining(" "))
                        .replace("\"", "\"\"");

                String exampleFiles = pattern.occs.stream()
                        .map(o -> sidToFileMap.get(o.docId))
                        .distinct()
                        .sorted()
                        .collect(Collectors.joining("|"));

                pw.printf("\"%s\",%d,%d,\"%s\"%n",
                        phrase, pattern.support, pattern.tokens.length, exampleFiles);
            }
        }
    }

    private static void processTextDirectory(String dirPath, String outputPath, int maxDocs) throws IOException {
        List<Path> files = Files.list(Paths.get(dirPath))
                .filter(p -> p.toString().endsWith(".txt")).sorted()
                .collect(Collectors.toList());

        if (maxDocs > 0 && files.size() > maxDocs) {
            files = files.subList(0, maxDocs);
        }

        int currentBitIndex = 0;
        List<Integer> docLengths = new ArrayList<>();
        int totalTokens = 0;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            for (int sid = 0; sid < files.size(); sid++) {
                sequencesSize.add(currentBitIndex);
                sidToFileMap.put(sid, files.get(sid).getFileName().toString());
                String content = new String(Files.readAllBytes(files.get(sid))).toLowerCase();
                String[] words = content.split("\\s+");
                int docLen = 0;
                for (String word : words) {
                    if (word.isEmpty())
                        continue;
                    docLen++;
                    int id = wordToIdMap.computeIfAbsent(word, k -> {
                        int newId = wordToIdMap.size() + 1;
                        idToWordMap.put(newId, k);
                        return newId;
                    });
                    writer.write(id + " -1 ");
                    currentBitIndex++;
                }
                writer.write("-2\n");
                docLengths.add(docLen);
                totalTokens += docLen;
            }
        }

        int numDocsProcessed = docLengths.size();
        int uniqueTokens = wordToIdMap.size();
        double avgDocLen = numDocsProcessed > 0 ? (double) totalTokens / numDocsProcessed : 0;
        int minDocLen = docLengths.stream().mapToInt(v -> v).min().orElse(0);
        int maxDocLen = docLengths.stream().mapToInt(v -> v).max().orElse(0);

        System.out.println("numDocsProcessed=" + numDocsProcessed);
        System.out.println("uniqueTokens=" + uniqueTokens);
        System.out.println("totalTokens=" + totalTokens);
        System.out.println("minDocLen=" + minDocLen);
        System.out.println("maxDocLen=" + maxDocLen);
        System.out.println("avgDocLen=" + String.format(Locale.US, "%.2f", avgDocLen));
    }

    private static void saveResultsToCSV(List<TreeSet<PatternVMSP>> results, String csvPath) throws Exception {
        if (results == null)
            return;

        Field bitmapField = PatternVMSP.class.getDeclaredField("bitmap");
        bitmapField.setAccessible(true);

        try (PrintWriter pw = new PrintWriter(new FileWriter(csvPath))) {
            pw.println("phrase,freq,length,example_files");
            for (TreeSet<PatternVMSP> level : results) {
                if (level == null)
                    continue;
                for (PatternVMSP pattern : level) {
                    String phrase = pattern.getPrefix().getItemsets().stream()
                            .flatMap(is -> is.getItems().stream())
                            .map(idToWordMap::get)
                            .collect(Collectors.joining(" "))
                            .replace("\"", "\"\"");

                    Bitmap bitmap = (Bitmap) bitmapField.get(pattern);
                    String exampleFiles = decodeSIDs(bitmap);

                    pw.printf("\"%s\",%d,%d,\"%s\"%n",
                            phrase, pattern.support, pattern.getPrefix().size(), exampleFiles);
                }
            }
        }
    }

    private static String decodeSIDs(Bitmap bitmap) {
        if (bitmap == null)
            return "";
        String sids = bitmap.getSIDs(sequencesSize);
        return Arrays.stream(sids.split(" "))
                .filter(s -> !s.isEmpty())
                .map(s -> sidToFileMap.get(Integer.parseInt(s)))
                .sorted()
                .collect(Collectors.joining("|"));
    }

    private static int countPatterns(List<TreeSet<PatternVMSP>> res) {
        if (res == null)
            return 0;
        return res.stream().filter(Objects::nonNull).mapToInt(Set::size).sum();
    }
}
