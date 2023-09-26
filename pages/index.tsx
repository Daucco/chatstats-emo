import Head from 'next/head'
import styles from '@/styles/Home.module.css'
import { useState, useEffect } from 'react';

import MsgBox from '@/components/msgbox';
import Predictor from '@/components/ml/predictor';
import Tokenizer from '@/components/ml/tokenizer';
import Vectorizer from '@/components/ml/vectorizer';
import Classifier from '@/components/ml/classifier';

import * as tf from '@tensorflow/tfjs';
import { LayersModel } from '@tensorflow/tfjs';

// Since vocabulary is defined in a local json this is legal,
//  but it can also be loaded with an async call in useEffect
import vocab from '../components/ml/models/tokenizer-vocab.json';

export default function Home() {
  const [message, setMessage] = useState('Write something awesome! :D'),
        [predictor, setPredictor] = useState<Predictor | null>(null),
        [predicted, setPredicted] = useState('');

  // Might be useful to declare tokenizer & vectorizer here, and then give them as arguments to Predictor
  //  allows more flexibility (eg, different tokenizers for different inputs, ...)
  useEffect(() => {
    // TODO: Load configs from another json file
    const minimodel_path = 'minimodel-tfjs/model.json';
    const max_vector_length = 100;

    const buildPredictor = async () =>{
      let tf_model: LayersModel;
      console.log("Loading TF model from " + minimodel_path);
      tf_model = await tf.loadLayersModel(minimodel_path);
      //console.log(tf_model);
  
      // TODO: Same for vectorizer and any other import
      const tok = new Tokenizer(),
            cla = new Classifier(tf_model, max_vector_length),
            vec = new Vectorizer(vocab, max_vector_length);

      setPredictor(new Predictor(tok, vec, cla));
      console.log("Predictor is ready!");
    }
    buildPredictor();
  }, []);

  // NOTE: predict method returns a list of prediction, one for each specified doc
  //  This is a dummy example, and there's only a single doc (the message)
  //let prediction: number[], sentiment: string;
  let prob: number, sentiment: string;

  function triggerPrediction(){
    // Raw message string as input
    if(predictor != null){
      prob = predictor.predict([message])[0];

      // TODO: Handle multiple outputs
      // Handles prediction result
      switch (Math.round(prob)) {
        case 0:
          sentiment = '💀💀';
          break;
        default:
          sentiment = '😀';
          break;
      }

      console.log(prob);
      
      // TODO: Hide predicted until there's a result
      setPredicted(sentiment);
    }
    else{
      setPredicted("Have a cow");
    }
  }

  return (
    <>
      <Head>
        <title>Chatstast - EMO module</title>
        <meta name="description" content="Barebones sentiment analyzer." />
      </Head>

      <main className={styles.main}>
        <h1>
          Sentiment analyzer
        </h1>

        <MsgBox
          message={message}
          setMessage={setMessage}
        />
        <button onClick={triggerPrediction}>
          How do you feel?
        </button>

        <p>Sentiment: {predicted}</p>
      </main>
    </>
  )
}