import { Parameters } from './parameters';
// TODO: TFJS

export class LogisticRegression {
  classes: number[];
  intercept: number[];
  values: number[][];

  constructor(params: Parameters) {
    this.classes = params.classes;
    this.intercept = params.intercept;
    this.values = params.values;
  }

  // Resolves probabilities for each class
  predict_proba(vector: number[]) {
    const b = (idx: number) => 1 / (1 + Math.exp(-this.func(vector, idx)));

    if (this.classes.length <= 2) {
      const result = b(0);
      return [1 - result, result];
    }
  }

  // intercept
  private func(vector: number[], forClass: number) {
    let sum = this.intercept[forClass];

    return vector.reduce((prev, vv, idx) => {
      const iv = vv * this.values[forClass][idx];
      return prev + iv;
    }, sum);
  }

  predict(vector: number[]): number {
    const probabilities = this.predict_proba(vector);
    const classes = this.classes;

    const stored = classes
      .reduce((prev, class_, idx) => {
        prev.push({
          class_,
          probability: probabilities[idx],
        });
        return prev;
      }, [])
      .sort((a, b) => b.probability - a.probability);

    return stored[0].class_;
  }
}
