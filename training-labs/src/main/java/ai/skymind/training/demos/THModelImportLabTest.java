package ai.skymind.training.demos;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by tomhanlon on 3/10/17.
 */
public class THModelImportLabTest {
    public static void main(String[] args) throws Exception{
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("/tmp/full_iris_model");

        INDArray myArray = Nd4j.zeros(1, 4); // one row 4 column array
        myArray.putScalar(0,0, 4.6);
        myArray.putScalar(0,1, 3.6);
        myArray.putScalar(0,2, 1.0);
        myArray.putScalar(0,3, 0.2);

        INDArray output = model.output(myArray);
        System.out.println("First Model Output");
        System.out.println(myArray);
        System.out.println(output);



    }
}
