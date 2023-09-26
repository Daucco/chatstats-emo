export default class Tokenizer{
    constructor(){
    }

    tokenize(text : string): string[]{
        text = text.toLowerCase();
        text = text.replace(/[!"#$%&()*+,-./:;<=>?@\[\\\]\^_`{|}~\t\n]/g, '');
        var tokens: string[] = text.split(' ');
        
        return tokens;
    }
}