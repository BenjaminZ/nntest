package translation;

import model.WordPair;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Created by Benjamin on 15/10/12.
 */
public class NNTrainer {
    public static void main(String[] args) {
        String sourceModelPath = "./source.txt";
        String targetModelPath = "./target.txt";
        int threshold = 10;
        WordVectors sourceModel = null;
        WordVectors targetModel = null;
        List<WordPair> wordPairList = new ArrayList<WordPair>();
        wordPairList.add(new WordPair("frowning", "frowning"));
        wordPairList.add(new WordPair("Reserve", "Reserve"));
        wordPairList.add(new WordPair("undermining", "undermining"));
        try {
            sourceModel = WordVectorSerializer.loadTxtVectors(new File(sourceModelPath));
            targetModel = WordVectorSerializer.loadTxtVectors(new File(targetModelPath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        int batchSize = 10;
        int seed = 123;

        MyDataSetIterator iter = new MyDataSetIterator(batchSize, threshold, new MyDataFetcher(sourceModel,
                targetModel, wordPairList));
        int outputNum = iter.totalOutcomes();
        int inputNum = iter.inputColumns();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .constrainGradientToUnitNorm(true)
                .learningRate(1e-6)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .useDropConnect(true)
                .list(1)
//                .layer(0, new DenseLayer.Builder()
//                        .nIn(inputNum)
//                        .nOut(100)
//                        .activation("relu")
//                        .weightInit(WeightInit.XAVIER)
//                        .build())
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(inputNum)
                        .nOut(outputNum)
                        .activation("identity")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
//        int showTimes = iterations / (iterations / 100);
        net.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        System.err.println("Train model...");
        for (int i = 0; i < 5; i++) {

            net.fit(iter);
        }
//        net.getLayer(0).setParam(DefaultParamInitializer.BIAS_KEY,Nd4j.zeros(1,100));

        System.out.println("Network parameters:");
        Map<String, INDArray> paras = net.getLayer(0).paramTable();
        System.out.println(net.params().toString());
        System.out.println("Weights:");
        System.out.println(paras.get("W"));
        System.out.println("Biases:");
        System.out.println(paras.get("b"));
        System.out.println(paras.get("b").length());
        String testSourceWord = "frowning";
        String testSourceWord2 = "Reserve";
        String testSourceWord3 = "undermining";
        INDArray testArray = sourceModel.getWordVectorMatrix(testSourceWord);
        INDArray testArray2 = sourceModel.getWordVectorMatrix(testSourceWord2);
        INDArray testArray3 = sourceModel.getWordVectorMatrix(testSourceWord3);
        System.out.println("Inputs:");
        System.out.println(testArray.toString());
        System.out.println(testArray2.toString());
        System.out.println(testArray3.toString());
        System.out.println("Outputs:");
        System.out.println(Arrays.toString(net.output(testArray).data().asFloat()));
        System.out.println(Arrays.toString(net.output(testArray2).data().asFloat()));
        System.out.println(Arrays.toString(net.output(testArray3).data().asFloat()));

//        try {
//            OutputStream fos = Files.newOutputStream(Paths.get("coefficients.bin"));
//            DataOutputStream dos = new DataOutputStream(fos);
//            Nd4j.write(net.params(), dos);
//            dos.flush();
//            dos.close();
//            FileUtils.write(new File("conf.json"), net.getLayerWiseConfigurations().toJson());
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

    }
}
