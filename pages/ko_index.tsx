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
        [predicted, setPredicted] = useState(''),

        // https://stackoverflow.com/questions/59125973/react-typescript-argument-of-type-is-not-assignable-to-parameter-of-type
        
        // Relevant:
        /*
            https://dev.to/yuikoito/face-detection-by-using-tensorflow-react-typescript-3dn5
            https://www.w3schools.com/react/react_useeffect.asp

            https://stackoverflow.com/questions/70979648/render-page-on-react-after-loading-json

            https://stackoverflow.com/questions/59125973/react-typescript-argument-of-type-is-not-assignable-to-parameter-of-type
            https://stackoverflow.com/questions/72208748/useeffect-promise-throwing-typescript-errors
        */
        [claModel, setClaModel] = useState<Classifier>();

  useEffect(() => {
    loadModels();
  }, []);
  
  const minimodel_path = 'minimodel-tfjs/model.json';
  //const sentimentDetect = async () =>{
  const loadModels = async () =>{
    const tf_model = await tf.loadLayersModel(minimodel_path);

    console.log(tf_model);
    setClaModel(new Classifier(tf_model));
    // Same for vectorizer and any other import
  }

  //sentimentDetect();
  

  

  // Right now this is useless, since every model loads a default config when instantiating
  // Might be usefull to declare tokenizer & vectorizer here, and then give them as arguments to Predictor
  //  allows more flexibility (eg, different tokenizers for different languages, ...)
  const tokenizer = new Tokenizer(),
        vectorizer = new Vectorizer();
        //classifier = new Classifier();

  
  //const classifier = new Classifier(claModel);
  
  const predictor = new Predictor(tokenizer, vectorizer, claModel);
  const labels = predictor.getLabels();
  let prediction: number[], sentiment: string;

  
  function triggerPrediction(){
    // Raw message string as input
    prediction = predictor.predict([postMsg]);

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
}import Head from 'next/head'
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
        [predicted, setPredicted] = useState(''),

        // https://stackoverflow.com/questions/59125973/react-typescript-argument-of-type-is-not-assignable-to-parameter-of-type
        [claModel, setClaModel] = useState<Classifier>();

  useEffect(() => {
    loadModels();
  }, []);
  
  const minimodel_path = 'minimodel-tfjs/model.json';
  //const sentimentDetect = async () =>{
  const loadModels = async () =>{
    const tf_model = await tf.loadLayersModel(minimodel_path);

    console.log(tf_model);
    setClaModel(new Classifier(tf_model));
    // Same for vectorizer and any other import
  }

  //sentimentDetect();
  

  

  // Right now this is useless, since every model loads a default config when instantiating
  // Might be usefull to declare tokenizer & vectorizer here, and then give them as arguments to Predictor
  //  allows more flexibility (eg, different tokenizers for different languages, ...)
  const tokenizer = new Tokenizer(),
        vectorizer = new Vectorizer();
        //classifier = new Classifier();

  
  //const classifier = new Classifier(claModel);
  
  const predictor = new Predictor(tokenizer, vectorizer, claModel);
  const labels = predictor.getLabels();
  let prediction: number[], sentiment: string;

  
  function triggerPrediction(){
    // Raw message string as input
    prediction = predictor.predict([postMsg]);

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