// NOTE: !!! This must match the same operations used when adjusting the imported models !!!
let _punctuationRegex = /[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g,
    _linkRegex = /(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)/g;


function _spellCheck(text){
    // TODO: TFJS
    return [];
}

function _tokenize(text){
    // TODO: TFJS
    return [];
}

function _lemmatize(wordList){
    // TODO: TFJS (lemmas or stems, whatever is faster)
    // Might be of use: https://medium.com/analytics-vidhya/building-a-stemmer-492e9a128e84
    return [];
}

class Cleaner {
    // Text preprocessing NLP module.
  
    constructor(){}
  
    clean(msg){
      // TODO: Implement operations:
      // 0. spell check
      // 1. substring subtract (links, handles, ...) + lowercase + punctuation removal (here of after everything else??)
      msg = msg.replace(_linkRegex, '');
      msg = msg.replace(_punctuationRegex, '');
  
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
export default Cleaner;
  