# I've tried to de-bias the Word Representation(For more info on this see at the top of the file named Debiasing Word Embeddings.ipyb). The code is in Python and uses Keras.
Note: All the files are well documented and commented wherever I felt necessary

The debiasing algorithm is from Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)

Run the file named Debiasing Word Embeddings.ipyb.ipynb which does the following:
1. Finds the similarity between the Word representations of 2 words
2. Gives the Analogy(see Debiasing Word Embeddings.ipyb)
3. Computes the gender Axis out of the 50 dimensions of our Word Representation
4. Neutralize bias for non-gender specific words like 'Receptionist'
5. Equalizes the gender-specific words like man, woman from say Baby-Sitter

