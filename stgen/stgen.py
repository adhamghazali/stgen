import torch as th
import torch.nn as nn
from transformers import T5EncoderModel
from stgen import configs
from stgen.configs import T5_CONFIGS
import string
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as f



MAX_LENGTH=256

class tokenizer:
    def __init__(self,model_name):

        self.model_name=model_name
        if self.model_name not in T5_CONFIGS.keys():
            print('model name is not found in config')

        self.config = T5_CONFIGS[self.model_name]

        if self.config[0] == 't5':
            t5_class, tokenizer_class = T5EncoderModel, T5Tokenizer

        elif self.config[0] == 'auto':
            t5_class, tokenizer_class = AutoModelForSeq2SeqLM, AutoTokenizer

        else:
            raise ValueError(f'unknown source {config[0]}')

        self.tokenizer = tokenizer_class.from_pretrained(model_name)

        #if th.cuda.is_available():
        #   self.t5_model = self.t5_model.cuda()
        #self.device = next(self.t5_model.parameters()).device
        #print("the device is", self.device)
        self.device= th.device('cuda')

    def get_tokens_masks(self,texts):

        encoded = self.tokenizer.batch_encode_plus(texts, return_tensors="pt",
                                              padding='max_length',
                                              max_length=MAX_LENGTH,
                                              truncation=False)

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        return input_ids,attn_mask


class stgen(th.nn.Module):

    def __init__(self,model_name,hidden_size,batch_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.embeddings_dims = configs.embedding_dims

        all_letters = string.ascii_letters + " .,;'<"
        self.output_size = len(all_letters) + 1  # Plus EOS marker

        self.t5 = T5EncoderModel.from_pretrained(model_name)

        #for param in self.t5.parameters():
            #param.requires_grad = False

        if th.cuda.is_available():
            self.t5 = self.t5.cuda()


        self.fc1=nn.Linear(configs.embedding_dims,configs.embedding_dims)


        self.i2h = nn.Linear(configs.embedding_dims + hidden_size, hidden_size)
        self.i2o = nn.Linear(configs.embedding_dims + hidden_size, self.output_size)
        self.o2o = nn.Linear(hidden_size + self.output_size, self.output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)



    def get_text_emb(self, tokens, mask):
        with th.no_grad():
            encoded_text = self.t5(input_ids=tokens, attention_mask=mask)['last_hidden_state'].float()
            #print('encoded_text shape', encoded_text.shape)
            embeddings = self.fc1(encoded_text[:, -1])
            #print('embeddings_shape before   ',embeddings.shape)



        return embeddings

    def rnn(self,embeddings,hidden):
        #print('concat shape', embeddings.shape, hidden.shape)
        input_combined = th.cat((embeddings, hidden), 1)

        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = th.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden



    def forward(self, tokens,mask, hidden):

        embeddings = self.get_text_emb(tokens, mask) # Transformer
        #print('before shape', embeddings.shape, hidden.shape)

        output, hidden=self.rnn(embeddings,hidden) #Character RNN

        return output, hidden



    def initHidden(self):

        weight = next(self.parameters()).data
        hidden = weight.new(self.batch_size, self.hidden_size).zero_().cuda()
        return hidden



