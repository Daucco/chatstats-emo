export class Cleaner {
  // Text preprocessing NLP module.

  // NOTE: !!! This must match the same operations used when adjusting the imported models !!!
  private punctuationRegex = /[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g;
  private linkRegex =
    /(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)/g;

  private spellCheck(text: string): string[] {
    // TODO: TFJS
    return [];
  }

  private tokenize(text: string): string[] {
    // TODO: TFJS
    return [];
  }

  private lemmatize(words: string[]): string[] {
    // TODO: TFJS (lemmas or stems, whatever is faster)
    // Might be of use: https://medium.com/analytics-vidhya/building-a-stemmer-492e9a128e84
    return [];
  }

  clean(msg: string): string[] {
    // TODO: Implement operations:
    // 0. spell check
    // 1. substring subtract (links, handles, ...) + lowercase + punctuation removal (here of after everything else??)
    msg = msg.replace(this.linkRegex, '');
    msg = msg.replace(this.punctuationRegex, '');

    const msg_words = msg
      .split(' ')
      .map((i) => i.toLowerCase())
      .filter((i) => i.length); // Skips empty words

    // 2. tokenizer
    // 3. stemmer
    // 4. stopwords
    return msg_words;
  }
}
