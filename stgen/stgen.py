import torch as th
import torch.nn as nn
from transformers import T5EncoderModel


class stgen(torch.nn.Module):

    def __init__(self,model_name,embeddings_dims,hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embeddings_dims = embeddings_dims

        all_letters = string.ascii_letters + " .,;'-"
        self.output_size = len(all_letters) + 1  # Plus EOS marker

        self.t5 = T5EncoderModel.from_pretrained(model_name)
        for param in self.t5.parameters():
            param.requires_grad = False


        self.i2h = nn.Linear(embeddings_dims + hidden_size, hidden_size)
        self.i2o = nn.Linear(embeddings_dims + hidden_size, self.output_size)
        self.o2o = nn.Linear(hidden_size + self.output_size, self.output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)



    def get_text_emb(self, tokens, mask):
        with th.no_grad():
            encoded_text = self.t5(input_ids=tokens, attention_mask=mask)['last_hidden_state'].float()
            outputs = self.fc1(encoded_text[:, -1])

        return outputs



    def forward(self, tokens,mask, hidden):
        text_outputs = self.get_text_emb(tokens, mask)

        input_combined = torch.cat(( text_outputs, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)


        return output, hidden

        def initHidden(self):
            return torch.zeros(1, self.hidden_size)



