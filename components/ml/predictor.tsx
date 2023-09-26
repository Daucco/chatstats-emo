// TODO: Import a json with essential configs (labels, ...)

import Tokenizer from "./tokenizer";
import Vectorizer from "./vectorizer";
import Classifier from "./classifier";

export default class Predictor{
    tokenizer: Tokenizer;
    vectorizer: Vectorizer;
    classifier: Classifier;

    constructor(tok: Tokenizer, vec: Vectorizer, cla: Classifier){
        this.tokenizer = tok;
        this.vectorizer = vec;
        this.classifier = cla;
    }
    predict(docs: string[]): number[]{
        let tokens: string[] = [];
        let doc_vector: number[][] = []; // Generates vectors of fixed length. If longer, vector is splitted.
        let docs_vector: number[][][] = [];
        let probabilities: number[] = [];
        let prob: number = 0;
        
        // Generates tokens for each doc
        docs.forEach(d =>{
            tokens = this.tokenizer.tokenize(d);
            doc_vector = this.vectorizer.vectorize(tokens);
            docs_vector.push(doc_vector);
        })

        // Gets predictions across multiple docs
        docs_vector.forEach(dv =>{
            prob = this.classifier.predict(dv);
            // TODO: Resolve prediction from prob


            probabilities.push(prob);
        })

        /*
        console.log("Raw input: " + docs);
        console.log("Tokens: " + tokens);
        console.log("Vector slices:" + doc_vector);
        console.log("Predictions: " + predictions);
        */

        return probabilities;
    }

    getLabels(): string[]{
        return ["not implemented"];
    }
}