import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaPreTrainedModel

class PhoBertLexical(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 768)
        self.dropout_nn = nn.Dropout(0.1)
        self.dropout_lm = nn.Dropout(0.1)
        self.init_weights()

    def forward(self, features, input_ids, attention_mask):
        x_nn = F.relu(self.fc1(features))
        x_nn = F.relu(self.fc2(x_nn))
        x_roberta = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        x_nn = self.dropout_nn(x_nn)
        x_roberta = self.dropout_lm(x_roberta)
        return torch.cat((x_nn, x_roberta), dim=1)

class PhobertContent(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout_llm = nn.Dropout(0.1)
        self.init_weights()

    def forward(self, input_ids_content, attention_mask_content):
        x_content_emb = self.roberta(input_ids=input_ids_content, attention_mask=attention_mask_content).last_hidden_state[:, 0, :]
        x_content_emb = self.dropout_llm(x_content_emb)
        return x_content_emb

class PhobertLexicalContent(nn.Module):
    def __init__(self, phobert_config, roberta_config):
        super(PhobertLexicalContent, self).__init__()
        self.num_classes = 2
        self.phobertlexical = PhoBertLexical(phobert_config)
        self.phobertContent = PhobertContent(roberta_config)
        self.out = nn.Linear(2304, 2)

    def forward(self, features, input_ids, attention_mask, input_ids_content, attention_mask_content, labels=None):
        x_lexical_emb = self.phobertlexical(features, input_ids=input_ids, attention_mask=attention_mask)
        x_content_emb = self.phobertContent(input_ids_content=input_ids_content, attention_mask_content=attention_mask_content)
        logits = self.out(torch.cat((x_lexical_emb, x_content_emb), dim=1))
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return {"loss": loss, "logits": logits}
        return type("obj", (object,), {"logits": logits})
