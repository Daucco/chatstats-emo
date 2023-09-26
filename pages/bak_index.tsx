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


export default function Home() {
  const [postMsg, setMsg] = useState('Write something awesome! :D'),
        [predicted, setPredicted] = useState('');

  // Right now this is useless, since every model loads a default config when instantiating
  // Might be usefull to declare tokenizer & vectorizer here, and then give them as arguments to Predictor
  //  allows more flexibility (eg, different tokenizers for different languages, ...)

  const [claModel, setClaModel] = useState<Classifier>();

const [predModel, setPred] = useState<Predictor | null>(null);

  useEffect(() => {
    const minimodel_path = 'minimodel-tfjs/model.json';
    const loadModels = async () =>{
      var tf_model;
      console.log(tf_model);
       tf_model = await tf.loadLayersModel(minimodel_path);
       console.log(tf_model);
  
      // Same for vectorizer and any other import
      
      //const cla = new Classifier(tf_model);
      const cla = new Classifier();

      // build predictor here and set it with usestate (initialize empty predictor)

      setClaModel(cla);
    }
    loadModels();
  }, []);

  
  //const sentimentDetect = async () =>{



  const tokenizer = new Tokenizer(),
        vectorizer = new Vectorizer(),
        //classifier = claModel;
        classifier = new Classifier();
  
  const predictor = new Predictor(tokenizer, vectorizer, classifier);
  const labels = predictor.getLabels();
  let prediction: number[], sentiment: string;

  
  function triggerPrediction(){
    // Raw message string as input
    //prediction = predictor.predict([postMsg]);
    const dummypred = predModel;

    if(dummypred != null){
      prediction = dummypred.predict([postMsg]);
    }

    // TODO: Handle multiple outputs

    // Handles prediction result
    switch (prediction[0]) {
      case 1:
        sentiment = '💀💀';
        break;
      default:
        sentiment = '😀';
        break;
    }
    
    // TODO: Hide predicted until there's a result
    setPredicted(sentiment);
    
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
        <p>Labels: {labels}</p>

        <MsgBox
          postMsg={postMsg}
          setMsg={setMsg}
        />
        <button onClick={triggerPrediction}>
          How do you feel?
        </button>

        <p>Sentiment: {predicted}</p>
      </main>
    </>
  )
}