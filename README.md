# I've tried to de-bias the Word Representation. The code is in Python and uses Keras.
Note: All the files are well documented and commented wherever I felt necessary
# The need for De-biasing:
Because word embeddings are very computationally expensive to train, most ML practitioners generally load a pre-trained set of embeddings.
But since even the Pre-Trained Word Embeddings are traied on some Text and the Text is written by humans, it captures the bias present in the author(c'mon) they're humans too!
But the problem arrises when out model learns the bias present in the texts it's trained on. Suppose such model is used for reviewing the Application of a Candidate for a Job, we don't want that it has biases like Gender, Race, etc.
In the Notebook I've tried to modify the Word Embeddings to reduce Gender Bias! Similarly we can do for other biases like Race,etc.

The debiasing algorithm is from Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)

Run the file named Debiasing Word Embeddings.ipyb.ipynb which does the following:
1. Finds the similarity between the Word representations of 2 words
2. Gives the Analogy(see Debiasing Word Embeddings.ipyb)
3. Computes the gender Axis out of the 50 dimensions of our Word Representation
4. Neutralize bias for non-gender specific words like 'Receptionist'
5. Equalizes the gender-specific words like man, woman from say Baby-Sitter

