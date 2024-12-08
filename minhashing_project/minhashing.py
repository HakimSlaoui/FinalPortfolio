import random

class MinHash:
    def __init__(self, num_hashes):
        self.num_hashes = num_hashes
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self):
        # Generate `num_hashes` hash functions
        max_shingle_id = 2**32 - 1  # Assume a large space for shingle IDs
        prime = 4294967311  # A prime number larger than max_shingle_id
        hash_functions = []
        for _ in range(self.num_hashes):
            a = random.randint(1, max_shingle_id)
            b = random.randint(0, max_shingle_id)
            hash_functions.append(lambda x, a=a, b=b: (a * x + b) % prime)
        return hash_functions

    def compute_signature(self, shingle_set):
        signature = []
        for h in self.hash_functions:
            min_hash = float("inf")
            for shingle in shingle_set:
                min_hash = min(min_hash, h(shingle))
            signature.append(min_hash)
        return signature

    @staticmethod
    def jaccard_similarity(sig1, sig2):
        assert len(sig1) == len(sig2), "Signatures must be of the same length."
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)


# Example Usage
if __name__ == "__main__":
    # Two sets of shingles
    set1 = {1, 3, 5, 7, 9}
    set2 = {3, 5, 7, 9, 11, 13}

    # Initialize MinHash with 100 hash functions
    minhash = MinHash(num_hashes=100)

    # Compute MinHash signatures
    sig1 = minhash.compute_signature(set1)
    sig2 = minhash.compute_signature(set2)

    # Estimate Jaccard similarity
    estimated_similarity = MinHash.jaccard_similarity(sig1, sig2)

    # Actual Jaccard similarity for comparison
    actual_similarity = len(set1 & set2) / len(set1 | set2)

    print(f"Estimated Jaccard Similarity: {estimated_similarity}")
    print(f"Actual Jaccard Similarity: {actual_similarity}")
