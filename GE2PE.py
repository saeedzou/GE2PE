from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
from normalizer import Normalizer

class GE2PE():

    def __init__(self, model_path = './content/checkpoint-320', GPU = False, dictionary = None):
        """ 
        model_path: path to where the GE2PE transformer is saved.
        GPU: boolean indicating use of GPU in generation.
        dictionary: a dictionary for self-defined words.
        """
        
        self.GPU = GPU
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        if self.GPU:
            self.model = self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dictionary = dictionary
        self.norma = Normalizer()
    
    def is_vowel(self, char):
        return (char in ['a', '/', 'i', 'e', 'u', 'o'])
   
    def rules(self, grapheme, phoneme):
        grapheme = grapheme.replace('آ', 'ءا')
        words = grapheme.split(' ')
        prons = phoneme.replace('1', '').split(' ')
        if len(words) != len(prons):
            return phoneme
        for i in range(len(words)):
            if 'ِ' not in words[i] and  'ُ' not in words[i] and 'َ' not in words[i]:
                continue
            for j in range(len(words[i])):
                if words[i][j] == 'َ':
                    if j == len(words[i]) - 1 and prons[i][-1] != '/':
                        prons[i] = prons[i] + '/'
                    elif self.is_vowel(prons[i][j]):
                        prons[i] = prons[i][:j] + '/' + prons[i][j+1:]
                    else:
                        prons[i] = prons[i][:j] + '/' + prons[i][j:]
                if words[i][j] == 'ِ':
                    if j == len(words[i]) - 1 and prons[i][-1] != 'e':
                        prons[i] = prons[i] + 'e'
                    elif self.is_vowel(prons[i][j]):
                        prons[i] = prons[i][:j] + 'e' + prons[i][j+1:]
                    else:
                        prons[i] = prons[i][:j] + 'e' + prons[i][j:]
                if words[i][j] == 'ُ':
                    if j == len(words[i]) - 1 and prons[i][-1] != 'o':
                        prons[i] = prons[i] + 'o'
                    elif self.is_vowel(prons[i][j]):
                        prons[i] = prons[i][:j] + 'o' + prons[i][j+1:]
                    else:
                        prons[i] = prons[i][:j] + 'o' + prons[i][j:]
        return ' '.join(prons)

    def lexicon(self, grapheme, phoneme):
        words = grapheme.split(' ')
        prons = phoneme.split(' ')
        output = prons
        for i in range(len(words)):
            try:
                output[i] = self.dictionary[words[i]]
                if prons[i][-1] == '1' and output[i][-1] != 'e':
                    output[i] = output[i] + 'e1'
                elif prons[i][-1] == '1' and output[i][-1] == 'e':
                    output[i] = output[i] + 'ye1'
            except:
              pass
        return ' '.join(output)

    def generate(self, input_list, batch_size = 10, use_rules = False, use_dict = False):
        """
        input_list: list of sentences to be phonemized.
        batch_size: inference batch_size
        use_rules: boolean indicating the use of rules to apply short vowels.
        use_dict: boolean indicating the use of self-defined dictionary.
        returns the list of phonemized sentences.
        """
        
        output_list = []
        input_list = [self.norma.normalize(text) for text in input_list]
        input = input_list
        input_list = [text.replace('ِ', '').replace('ُ', '').replace('َ', '') for text in input_list]
        for i in range(0,len(input_list),batch_size):
            in_ids = self.tokenizer(input_list[i:i+batch_size], padding=True,add_special_tokens=False, return_attention_mask=True,return_tensors='pt')
            if self.GPU:
                out_ids = self.model.generate(in_ids["input_ids"].cuda(), attention_mask=in_ids["attention_mask"].cuda(), num_beams=5,
                                        min_length= 1, max_length=512, early_stopping=True,)
            else:
                out_ids = self.model.generate(in_ids["input_ids"], attention_mask=in_ids["attention_mask"], num_beams=5,
                                        min_length= 1, max_length=512, early_stopping=True,)
            output_list += self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        
        
        if use_dict:
            for i in range(len(input_list)):
                output_list[i] = self.lexicon(input_list[i], output_list[i])

        if use_rules:
            for i in range(len(input_list)):
                output_list[i] = self.rules(input[i], output_list[i])

        output_list = [i.strip().replace("1", "") for i in output_list]
        return output_list

    def generate_with_punctuation(self, input_list, batch_size=10, use_rules=False, use_dict=False):
        """
        Generates phonemes for sentences containing punctuation.
        Splits sentences by punctuation, generates phonemes for each part, and then joins them.
        input_list: list of sentences or a single sentence string.
        batch_size: inference batch_size
        use_rules: boolean indicating the use of rules to apply short vowels.
        use_dict: boolean indicating the use of self-defined dictionary.
        returns the list of phonemized sentences.
        """
        if isinstance(input_list, str):
            input_list = [input_list]

        puncts = ".!؟،:"
        
        all_segments = []
        sentence_structures = []
        
        for text in input_list:
            # Split by punctuation and keep the delimiters
            parts = [p.strip() for p in re.split(f'([{puncts}])', text) if p.strip()]
            sentence_structures.append(parts)
            
            # Collect only the text parts for the model
            for part in parts:
                if part not in puncts:
                    all_segments.append(part)

        # Generate phonemes for all text segments at once
        if all_segments:
            phonemized_segments = self.generate(all_segments, batch_size, use_rules, use_dict)
        else:
            phonemized_segments = []

        result_list = []
        phoneme_idx = 0
        for structure in sentence_structures:
            result_sentence = ""
            for part in structure:
                if part in puncts:
                    # For punctuation, remove preceding space, then add the punctuation and a space.
                    result_sentence = result_sentence.strip() + part + " "
                else:
                    # For text, add the phonemized part and a space.
                    if phoneme_idx < len(phonemized_segments):
                        result_sentence += phonemized_segments[phoneme_idx] + " "
                        phoneme_idx += 1
            result_list.append(result_sentence.strip())
            
        return result_list
