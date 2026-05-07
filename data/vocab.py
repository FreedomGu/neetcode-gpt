from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        text_sorted = sorted(text)
        stoi = {}
        itos = {}
        integer = 0
        for i in text_sorted:
            if i in stoi:
                continue
            stoi[i] = integer
            itos[integer] = i
            integer += 1
        #print(stoi, itos)
        return stoi, itos
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        out = []
        #for i in text:
        #    out.append(self.stoi[i])
        #out.append(stoi[i] for i in text)
        return [stoi[ch] for ch in text]#out
        #pass

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        #pass
        out = ""
        #for i in ids:
        #    out += itos[i]
        return ''.join(itos[i] for i in ids)
