export class Vectorizer {
  // NOTE: Any proccessing operation must match that of the imported vectorizer
  private punctuationRegex = /[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g;
  private words: string[];

  constructor(params: string[]) {
    // Imports vocabulary from external configuration
    this.words = params;
  }

  transform(msg: string): number[] {
    msg = msg.replace(this.punctuationRegex, '');

    // NOTE: Preprocessing. This must be consistent with the imported models
    const msg_words = msg
      .split(' ')
      .map((i) => i.toLowerCase())
      .filter((i) => i.length); // Skips empty words

    // Maps words to vector form.
    // Count vectorize approach:
    // Replaces corresponding word in vector with the number of occurrences in the message
    const vector = this.words.map(
      (word) => msg_words.filter((w) => w === word).length
    );

    return vector;
  }
}
