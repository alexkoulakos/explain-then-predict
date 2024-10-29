import torch
import torch.nn as nn

from transformers import EncoderDecoderModel, AutoTokenizer

from models.expl_to_label import Expl2LabelModel

class ExplainThenPredictModel(nn.Module):
    def __init__(self, args):
        super(ExplainThenPredictModel, self).__init__()

        self.device = args.device
        self.encoder_max_len = args.encoder_max_len
        self.generation_config = args.generation_config

        ## Load pretrained models

        # Load Seq2Seq model
        self.seq2seq_model = EncoderDecoderModel.from_pretrained(args.seq2seq_model).to(self.device)

        # Load Expl2Label model
        self.expl2label_model = Expl2LabelModel(args.expl2label_encoder_checkpt).to(self.device)
        
        if self.device == torch.device('cpu'):
            self.expl2label_model.load_state_dict(torch.load(args.expl2label_model, map_location=self.device)['model_state_dict'])
        else:
            self.expl2label_model.load_state_dict(torch.load(args.expl2label_model)['model_state_dict'])

        ## Load intermediate tokenizers

        # Tokenizer for Seq2Seq decoder part
        self._seq2seq_decoder_tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model.config.decoder._name_or_path)

        # Tokenizer for Expl2Label task
        self._expl2label_tokenizer = AutoTokenizer.from_pretrained(args.expl2label_encoder_checkpt)

    def _batch_decode(self, pred_expls_tokens):
        return self._seq2seq_decoder_tokenizer.batch_decode(pred_expls_tokens, skip_special_tokens=True)
    
    def _encode(self, pred_expls):
        encoder_input = self._expl2label_tokenizer(
            pred_expls,
            padding="max_length",
            max_length=self.encoder_max_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        input_ids = encoder_input['input_ids'].to(self.device)
        attention_mask = encoder_input['attention_mask'].to(self.device)

        return input_ids.to(self.device), attention_mask.to(self.device)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Seq2Seq model (generate explanations)
        pred_expls_tokens = self.seq2seq_model.generate(
            input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=self.seq2seq_model.config.decoder_start_token_id,
            eos_token_id=self.seq2seq_model.config.eos_token_id,
            pad_token_id=self.seq2seq_model.config.pad_token_id,
            generation_config=self.generation_config
        )

        pred_expls = self._batch_decode(pred_expls_tokens)

        input_ids, attention_mask = self._encode(pred_expls)
        
        # Expl2Label model (predict label based on explanations)
        pred_labels = self.expl2label_model(input_ids, attention_mask)

        return pred_expls, pred_labels