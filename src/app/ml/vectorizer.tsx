export class Vectorizer {
  private words: string[];

  constructor(params: string[]) {
    // Imports vocabulary from external configuration
    this.words = params;
  }

  // Transforms input string into a number vector (BoW)
  transform(msg_words: string[]): number[] {
    // TODO: TF-IDF
    const vector = this.words.map(
      (word) => msg_words.filter((w) => w === word).length
    );

    return vector;
  }
}
