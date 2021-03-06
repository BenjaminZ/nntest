package translation;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * My simple Iterator using my DataFetcher.
 * Created by Benjamin on 15/10/12.
 */
public class MyDataSetIterator extends BaseDatasetIterator {
    public MyDataSetIterator(int batch, int numExamples, MyDataFetcher fetcher) {
        super(batch, numExamples, fetcher);
    }
}
