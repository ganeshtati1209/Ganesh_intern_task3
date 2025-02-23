import random

class MarkovChainTextGenerator:
    """
    A simple text generation model using Markov Chains.
    Trains on input text and generates sentences based on learned probabilities.
    """

    def __init__(self, n=2):
        self.n = n  # Order of Markov Chain (number of previous words used)
        self.markov_dict = {}

    def train(self, text):
        """
        Trains the Markov model using the provided text.
        """
        words = text.split()
        for i in range(len(words) - self.n):
            key = tuple(words[i:i + self.n])  # Previous words as key
            next_word = words[i + self.n]  # Word to predict
            self.markov_dict.setdefault(key, []).append(next_word)

    def generate_text(self, seed, length=20):
        """
        Generates text using the trained Markov model.
        """
        seed_words = seed.split()
        if len(seed_words) < self.n:
            raise ValueError(f"Seed must contain at least {self.n} words.")

        result = seed_words[:]
        for _ in range(length):
            key = tuple(result[-self.n:])
            next_words = self.markov_dict.get(key, None)
            if not next_words:
                break  # Stop if no possible words
            next_word = random.choice(next_words)
            result.append(next_word)

        return " ".join(result)


# Example usage
if __name__ == "__main__":
    # Sample training text
    training_text = """
    The sun rises in the east and sets in the west. The sky is blue, and the grass is green.
    Nature is beautiful and full of wonders.
    """


    generator = MarkovChainTextGenerator(n=2)
    generator.train(training_text)


    seed_text = "The sky"
    generated_text = generator.generate_text(seed_text, length=15)

    print("\nGenerated Text:\n", generated_text)
