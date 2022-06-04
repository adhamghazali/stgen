import torch as th
import torch.nn as nn
from transformers import T5EncoderModel


class stgen(torch.nn.Module):

    def __init__(self,model_name):
        super().__init__()

        self.t5 = T5EncoderModel.from_pretrained(model_name)

        for param in self.t5.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512, 1024) # 512 if using t5-small
        #self.some_layer_tbd


    def get_text_emb(self, tokens, mask):
        with th.no_grad():
            encoded_text = self.t5(input_ids=tokens, attention_mask=mask)['last_hidden_state'].float()
            outputs = self.fc1(encoded_text[:, -1])

        return outputs

    def forward(self, tokens,mask):

        text_outputs = self.get_text_emb(tokens, mask)
        x=self.some_layer_tbd(text_outputs)

        return x
