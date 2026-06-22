import random
from corruptions.typo import typo_noise
from corruptions.dropout import word_dropout
from corruptions.delete import delete_char

class BenchmarkCorruptor:
    def corrupt(self, query: str, tier: str) -> str:
        tier = tier.lower()
        
        if tier == "easy":
            if random.random() > 0.5:
                return typo_noise(query)
            return word_dropout(query, p=0.1)

        elif tier == "medium":
            text = delete_char(query)
            return typo_noise(text)

        elif tier == "hard":
            text = word_dropout(query, p=0.35)
            text = typo_noise(text)
            return delete_char(text)
            
        return query
