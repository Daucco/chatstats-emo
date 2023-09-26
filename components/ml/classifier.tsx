import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';

export default class Classifier{
    model: LayersModel;
    max_vector_length: number;

    constructor(tf_model: LayersModel, max_vector_length: number){
        this.model = tf_model;
        this.max_vector_length = max_vector_length;
    }

    predict(doc_vector: number[][]): number{
        // Computes class probabilities from a sliced document of tokens

        // Positional: values, shape, dtype. See: https://js.tensorflow.org/api/latest/#tensor
        let tf_vector: tf.Tensor = tf.tensor2d(doc_vector, [doc_vector.length, this.max_vector_length], 'int32');
    
        // Gets prediction as array of probabilities (one per doc slice)
        // keep in mind: https://stackoverflow.com/questions/64211494/how-to-use-predict-in-tensorflow-js
        let x_probs = this.model.predict(tf_vector) as tf.Tensor;

        // Averages probabilities and transforms output into a nested array which resolves to a single value (?)
        //const mean_x = tf.mean(x).arraySync();
        const mean_x_probs = tf.mean(x_probs).dataSync()[0];

        return mean_x_probs;
    }
}