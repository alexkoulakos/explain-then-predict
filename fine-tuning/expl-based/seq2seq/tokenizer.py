from transformers import (
    AutoTokenizer, 
    GPT2Tokenizer, 
    BatchEncoding
)
from typing import List

class Seq2SeqTokenizer():
    def __init__(self, encoder_checkpt, decoder_checkpt, encoder_max_len, decoder_max_len):
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len

        # Encoder tokenizer configurations
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_checkpt)
        self.encoder_tokenizer.bos_token = self.encoder_tokenizer.cls_token
        self.encoder_tokenizer.eos_token = self.encoder_tokenizer.sep_token

        # Decoder tokenizer configurations
        def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
            outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            
            return outputs

        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        self.decoder_tokenizer = GPT2Tokenizer.from_pretrained(decoder_checkpt)
        self.decoder_tokenizer.pad_token = self.decoder_tokenizer.unk_token
    
    def encode_premises_and_hypotheses(self, premises, hypotheses) -> BatchEncoding:
        return self.encoder_tokenizer(
            premises, 
            hypotheses, 
            padding='max_length', 
            max_length=self.encoder_max_len, 
            truncation=True, 
            add_special_tokens=True,
            return_tensors='pt'
        )

    def encode_explanations(self, explanations) -> BatchEncoding:
        return self.decoder_tokenizer(
            explanations,
            padding='max_length', 
            max_length=self.decoder_max_len, 
            truncation=True, 
            add_special_tokens=True,
            return_tensors='pt'
        )
    
    def decode(self, input_ids: List[int], gpt2=True) -> str:
        if gpt2:
            return self.decoder_tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            return self.encoder_tokenizer.decode(input_ids, skip_special_tokens=True)
    
    def batch_decode(self, input_ids: List[List[int]], gpt2=True) -> List[str]:
        if gpt2:
            return self.decoder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        else:
            return self.encoder_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    
    def __call__(self, text, gpt2=False):
        if not gpt2:
            assert isinstance(text, tuple), "Both premise and hypothesis need to be passed to BERT Tokenizer"

            premises, hypotheses = text

            return self.encode_premises_and_hypotheses(premises, hypotheses)
        else:
            assert isinstance(text, list), "Explanation only needs to be passed to GPT Tokenizer"

            return self.encode_explanations(text)