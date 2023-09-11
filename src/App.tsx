import { Predictor } from './app/ml/predictor';
import { useState } from 'react';

import './style.css';

export const App = () => {
  const [postMsg, setMsg] = useState('Write something awesome! :D'),
    [predicted, setPredicted] = useState(''),
    predictor = new Predictor(),
    labels = predictor.getLabels();

  let prediction: number, sentiment: string;

  function handleSubmit(e) {
    e.preventDefault();

    // Takes raw message string as input
    prediction = predictor.predict(postMsg);

    // Handles prediction result
    switch (prediction) {
      case 1:
        sentiment = '💀💀';
        break;
      default:
        sentiment = '😀';
        break;
    }
    setPredicted(sentiment);
  }

  return (
    <div>
      <h1>Sentiment analyzer</h1>
      <p>Labels: {labels.join(', ')}</p>

      <form method="post" onSubmit={handleSubmit}>
        <textarea
          rows={3}
          cols={30}
          value={postMsg}
          onChange={(e) => setMsg(e.target.value)}
        />
        <button type="submit">Check sentiment!</button>
      </form>

      <p>Sentiment: {predicted}</p>
    </div>
  );
};
