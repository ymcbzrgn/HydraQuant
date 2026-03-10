import numpy as np

class BinaryQuantizer:
    """
    Implements Binary Quantization for dense embeddings to accelerate retrieval.
    Converts float32 embeddings into bit-packed uint8 arrays and computes Hamming distance.
    This provides up to 32x memory reduction and significantly faster distance calculations
    with minimal (>99%) retrieval quality loss.
    """
    
    @staticmethod
    def binarize(embeddings: np.ndarray) -> np.ndarray:
        """Converts float arrays to binary (0/1) based on > 0 threshold."""
        # Handles both 1D and 2D arrays
        return (embeddings > 0).astype(np.int8)

    @staticmethod
    def pack(binary_embeddings: np.ndarray) -> np.ndarray:
        """Packs binary (0/1) arrays into 8-bit integers."""
        return np.packbits(binary_embeddings, axis=-1)

    @staticmethod
    def binarize_and_pack(embeddings: np.ndarray) -> np.ndarray:
        """Helper to do thresholding and packing in one step."""
        return BinaryQuantizer.pack(BinaryQuantizer.binarize(embeddings))

    @staticmethod
    def hamming_distance(packed_query: np.ndarray, packed_docs: np.ndarray) -> np.ndarray:
        """
        Calculates Hamming distance using bitwise XOR and popcount.
        packed_query shape: (dim,)
        packed_docs shape: (N, dim)
        """
        # XOR bits so that 1s indicate mismatches
        xor_result = np.bitwise_xor(packed_docs, packed_query)
        
        # Unpack back to 0/1 to count the mismatches (popcount)
        unpacked = np.unpackbits(xor_result, axis=-1)
        
        # Sum along the dimension axis to get hamming distance
        distances = np.sum(unpacked, axis=-1)
        return distances
