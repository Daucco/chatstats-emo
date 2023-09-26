

export default class Vectorizer{
    vocabulary: {[key: string]: number};
    max_vector_length: number;
    
    constructor(vocab: {[key: string]: number} ,max_vector_length: number){
        this.vocabulary = vocab;
        this.max_vector_length = max_vector_length;
    }


    vectorize(tokens : string[]){
        var vect : number[] = [];
        const mvl = this.max_vector_length;

        // Resolves vectorized form out of tokens
        tokens.forEach(term => {
            if(this.vocabulary[term] != undefined){
                vect.push(this.vocabulary[term]);
            }
        })

        // Creates slices out of generated vector
        // This must match the vector length expected by the classifier
        const VECTOR_LENGTH: number = vect.length;
        //const MAX_VECTOR_LENGTH: number = 100;
        const HALF_VECTOR: number = Math.trunc(mvl / 2);

        var vector_slices: number[][] = [];

        let i = 0;
        while(i+HALF_VECTOR < Math.max(VECTOR_LENGTH, mvl)){
            var slice: number[] = vect.slice(i, i+mvl);

            // TODO: Make something better, this is so dirty. This just deals with padding.
            while(slice.length < mvl){
                slice.push(0);
            }
            vector_slices.push(slice);

            // Slices contains tokens from adjacent slices
            i = i + HALF_VECTOR;
        }

        return vector_slices;
    }
}