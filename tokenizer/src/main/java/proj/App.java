package proj;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.util.*;

/**
 * Hello world!
 */
public class App {
    public static String inputPath = "europarl-v7.de-en.en";
    public static String outputPath = "europarl-v7.de-en.en.tok";

    public static void main(String[] args) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize");
        props.setProperty("tokenize.language", "English");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // tokenize a very large file, one line at a time
        // and write the tokenized output to a new file
        try (BufferedReader br = new BufferedReader(new FileReader(inputPath));
                BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.toLowerCase();
                CoreDocument doc = new CoreDocument(line);
                pipeline.annotate(doc);
                List<CoreLabel> tokens = doc.tokens();
                for (CoreLabel token : tokens) {
                    bw.write(token.word());
                    bw.write(" ");
                }
                bw.newLine();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
