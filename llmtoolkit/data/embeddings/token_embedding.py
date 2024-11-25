from typing import List, Tuple, Union

import torch


class TokenEmbedding:
    """TokenEmbedding class for loading and using pre-trained word embeddings
    like GloVe or fastText."""

    def __init__(self, file_name: str):
        """Initialize the TokenEmbedding instance.

        Args:
            file_name (str): Path to the pre-trained word embedding model file.
        """
        self.tokens, self.embeddings = self.load_embedding(file_name)
        self.token_to_idx = {
            token: idx
            for idx, token in enumerate(self.tokens)
        }

    def load_embedding(self, file_name: str) -> Tuple[List[str], torch.Tensor]:
        """Load the pre-trained word embeddings from a file.

        Args:
            file_name (str): Path to the pre-trained word embedding model file.

        Returns:
            Tuple[List[str], torch.Tensor]: List of tokens and a tensor of word vectors.
        """
        tokens, embeds = [], []

        # Load the word embeddings from the file
        with open(file_name,
                  'r',
                  encoding='utf-8',
                  newline='\n',
                  errors='ignore') as f:
            for line in f.readlines():
                line = line.rstrip().split(' ')
                token, elems = line[0], list(map(float, line[1:]))
                if len(elems) > 1:
                    tokens.append(token)
                    embeds.append(elems)
        return tokens, torch.tensor(embeds, dtype=torch.float)

    def get_word_embedding(self, tokens: Union[str,
                                               List[str]]) -> torch.Tensor:
        """Get word embeddings for one or more tokens.

        Args:
            tokens (str or List[str]): Token(s) for which to retrieve embeddings.

        Returns:
            torch.Tensor: Word embeddings as a tensor.
        """
        if isinstance(tokens, str):
            tokens = [tokens]

        indices = [self.token_to_idx.get(token) for token in tokens]
        vecs = self.embeddings[torch.tensor(indices)]
        return vecs

    def vocabulary_size(self) -> int:
        """Get the size of the vocabulary, including the '<unk>' token.

        Returns:
            int: Size of the vocabulary.
        """
        return len(self.tokens)

    def cosine_similarity(self, embeddings_mat: torch.Tensor,
                          query_vector: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two tensors.

        Args:
            embeddings_mat (torch.Tensor): Matrix of word embeddings.
            query_vector (torch.Tensor): Query vector for which to find cosine similarities.

        Returns:
            torch.Tensor: Cosine similarities between the query vector and all vectors in the matrix.
        """
        # Normalize the query vector
        normed_query = query_vector / torch.norm(query_vector)
        # Normalize all embeddings in the matrix
        normed_embeddings = embeddings_mat / torch.norm(
            embeddings_mat, dim=1, keepdim=True)
        # Calculate cosine similarities
        similarity_scores = torch.matmul(normed_embeddings, normed_query.T)
        return similarity_scores

    def get_knn_neighbors(self,
                          query_vector: torch.Tensor,
                          top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Get the K-nearest neighbors based on cosine similarities between
        word vectors.

        Args:
            query_vector (torch.Tensor): Query vector for which to find nearest neighbors.
            top_k (int, optional): Number of nearest neighbors to retrieve. Default is 5.

        Returns:
            Tuple[List[str], List[float]]: List of K-nearest neighbor words and their cosine distances.
        """
        similarity_scores = self.cosine_similarity(self.embeddings,
                                                   query_vector)
        # Get the indices of the K-nearest neighbors (excluding the query word itself)
        topk_indices = similarity_scores.argsort(dim=0,
                                                 descending=True)[1:top_k + 1]
        # Get the tokens corresponding to the indices
        topk_words = [self.tokens[idx] for idx in topk_indices]
        # Get the corresponding cosine distances
        topk_distance = [similarity_scores[i].item() for i in topk_indices]
        return topk_words, topk_distance

    def find_k_nearest_neighbors(self,
                                 query_word: str,
                                 top_k: int = 5) -> List[str]:
        """Find the K-nearest neighbors for a query word based on cosine
        similarities between word vectors.

        Args:
            query_word (str): The word for which to find nearest neighbors.
            top_k (int, optional): The number of nearest neighbors to retrieve. Default is 5.

        Returns:
            List[str]: A list of K-nearest neighbor words.
        """
        if query_word not in self.token_to_idx:
            return []
        # Return an empty list if the query word is not in the vocabulary
        # Get the embedding for the query word
        query_vector = self.get_word_embedding(query_word)
        # Get the indices and distances of the k-nearest neighbors
        topk_words, topk_distance = self.get_knn_neighbors(query_vector,
                                                           top_k=top_k)

        return topk_words, topk_distance

    def get_analogy_words(self,
                          word1: str,
                          word2: str,
                          word3: str,
                          k: int = 1) -> List[str]:
        """
        Get words that are similar to the analogy: word1 - word2 + word3.

        Args:
            word1 (str): First word in the analogy.
            word2 (str): Second word in the analogy.
            word3 (str): Third word in the analogy.
            k (int, optional): The number of similar words to retrieve. Default is 1.

        Returns:
            List[str]: A list of words that are similar to the given analogy.
        """
        vecs = self.get_word_embedding([word1, word2, word3])
        unk_vec = vecs[1] - vecs[0] + vecs[2]
        unk_vec = unk_vec.reshape(vecs[0].shape)
        knn_words, _ = self.get_knn_neighbors(unk_vec, k)
        return knn_words


if __name__ == '__main__':
    # Example usage:
    embedding = TokenEmbedding(
        '/home/robin/work_dir/llm/nlp-toolkit/examples/language_model/glove.vec'
    )
    query_word = 'beautiful'
    similar_words, similar_distance = embedding.find_k_nearest_neighbors(
        query_word, top_k=5)
    print(
        f"Words similar to '{query_word}': {similar_words}: distances: {similar_distance}"
    )

    # Example usage:
    analogy_word = embedding.get_analogy_words('man', 'woman', 'king')
    print(analogy_word)
