package model;

/**
 * Stores word pair information.
 * Created by Benjamin on 15/9/24.
 */
public class WordPair {
    public WordPair(String sourceWord, String targetWord) {
        this.sourceWord = sourceWord.toLowerCase();
        this.targetWord = targetWord.toLowerCase();
        this.count = 1;
    }

    private String sourceWord;
    private String targetWord;
    private int count;

    public int getCount() {
        return count;
    }

    public void countAdd() {
        count++;
    }

    public void resetCount() {
        count = 0;
    }

    public String getSourceWord() {
        return sourceWord;
    }

    public String getTargetWord() {
        return targetWord;
    }

    public String getKey() {
        return sourceWord;
    }
}
