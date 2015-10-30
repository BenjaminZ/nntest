package translation;

import model.WordPair;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

/**
 * My simple data fetcher. Put the INDArray from the source word and the INDArray from the target word into a DataSet.
 * The INDArrays are from the sourceModel and targetModel, which are the Word2Vec models.
 * Created by Benjamin on 15/10/12.
 */
public class MyDataFetcher extends BaseDataFetcher {
    private WordVectors sourceModel;
    private WordVectors targetModel;
    private List<WordPair> wordPairList;

    public MyDataFetcher(WordVectors sourceModel, WordVectors targetModel, List<WordPair> wordPairList) {
        this.sourceModel = sourceModel;
        this.targetModel = targetModel;
        this.wordPairList = wordPairList;
        this.numOutcomes = 100;
        this.inputColumns = 100;
        this.totalExamples = wordPairList.size();
    }

    public void fetch(int numExamples) {
        List<DataSet> list = new ArrayList<DataSet>();
        int from = this.cursor;
        int to = this.cursor + numExamples;
        if (to > this.totalExamples) {
            to = this.totalExamples;
        }
        WordPair wp;
        for (int i = from; i < to; i++) {
            wp = wordPairList.get(i);
            INDArray sourceVector = sourceModel.getWordVectorMatrix(wp.getSourceWord());
            INDArray targetVector = targetModel.getWordVectorMatrix(wp.getTargetWord());
            DataSet add = new DataSet(sourceVector, targetVector);
            list.add(add);
        }
        initializeCurrFromList(list);
        this.cursor += numExamples;
    }
}
