import { Parameters } from './parameters';
import { Vectorizer } from './vectorizer';
import { LogisticRegression } from './classifier';

import * as variables from './baseline_configs.json';

export class Predictor {
  params: Parameters;
  vectorizer: Vectorizer;
  logReg: LogisticRegression;

  constructor() {
    // Imports configurations and initializes models
    this.params = (variables && variables['default']) || (variables as any);
    this.vectorizer = new Vectorizer(this.params.words);
    this.logReg = new LogisticRegression(this.params);
  }

  getLabels() {
    return this.params.labels;
  }

  // Fires vectorizer and classifier
  // Returns predicted class
  predict(msg: string): number {
    const vector = this.vectorizer.transform(msg);
    //return this.logReg.predict(vector) === 1;

    return this.logReg.predict(vector);
  }
}
