import { Parameters } from './parameters';
import { Cleaner } from './cleaner';
import { Vectorizer } from './vectorizer';
import { LogisticRegression } from './classifier';

import * as variables from './baseline_configs.json';

export class Predictor {
  params: Parameters;
  cleaner: Cleaner;
  vectorizer: Vectorizer;
  logReg: LogisticRegression;

  constructor() {
    // Imports configurations and initializes models
    this.params = (variables && variables['default']) || (variables as any);
    this.cleaner = new Cleaner();
    this.vectorizer = new Vectorizer(this.params.words);
    this.logReg = new LogisticRegression(this.params);
  }

  getLabels() {
    return this.params.labels;
  }

  // Fires vectorizer and classifier
  // Returns predicted class
  predict(msg: string): number {
    const cleanMsg = this.cleaner.clean(msg);
    const vector = this.vectorizer.transform(cleanMsg);
    return this.logReg.predict(vector);
  }
}
