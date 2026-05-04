from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        corpus_list = list(corpus)
        # dict = save frequences
        output = []
        for i in range(num_merges):
            if len(corpus_list) < i:
                break
            pairs = {}
            for j in range(len(corpus_list)-1):
                pair = (corpus_list[j], corpus_list[j+1])
                pairs[pair] = pairs.get(pair, 0) + 1
            best = max(pairs.values()) # ab bc ca
            #print(best)
            res = sorted([pair for pair in pairs.keys() if pairs[pair] == best])
            output.append(res[0])
            # merge all non overlapping occurrences left to right:
            new_list = []
            k = 0 
            while k < len(corpus_list):
                if corpus_list[k]==res[0][0] and corpus_list[k+1]==res[0][1]:
                    k = k + 2
                    new_list.append(res[0][0]+res[0][1])
                else:
                    new_list.append(corpus_list[k])
                    k = k+1
            corpus_list = new_list.copy()
        return output
                    
        # for each steps : 
            # count frequency 
            # find the most frequent pair
            # merge all non-overlapping occurrenes 
            # record the merge
        #for i in  for to get dict frequency

        #find  pairs and merge 
        # 
        #pass
