from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import masked_cross_entropy



class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights=params['weights']
        self.train_att= params['train_att']
        self.lam = params['att_lambda']
        self.num_sv_heads=params['num_supervised_heads']
        self.sv_layer = params['supervised_layer_pos']
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.softmax=nn.Softmax(config.num_labels)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        #logits = self.softmax(logits)
        print("----------------- FORWARD DEBUG START ")
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # in your training loop, immediately after unpacking the batch:

        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            if(self.train_att):
                
                loss_att=0
                for i in range(self.num_sv_heads):
                    attention_weights=outputs[1][self.sv_layer][:,i,0,:]

                    # === DEBUG PRINTS ===
                    # print batch-summary stats so you can see if anything is all zeros
                    print(f"[ DEBUG] head {i} attention min/max/sum:",
                          attention_weights.min().item(),
                          attention_weights.max().item(),
                          attention_weights.sum(dim=1).tolist())
                    # also check your provided attention_vals
                    print(f"[ DEBUG] provided attention_vals min/max/sum:",
                          attention_vals.min().item(),
                          attention_vals.max().item(),
                          attention_vals.sum(dim=1).tolist())
                    # optionally assert that we’re not all zeros
                    if attention_weights.sum().item() == 0:
                        print(f"[ WARNING] head {i} produced ALL-zero attention for this batch!")

                    loss_att +=self.lam*masked_cross_entropy(attention_weights,attention_vals,attention_mask)
                    if (attention_vals != 0 ).any(): print("NON zero attention_vals")
                loss = loss + loss_att
            outputs = (loss,) + outputs
        print("----------------- FORWARD DEBUG  END")


        return outputs  # (loss), logits, (hidden_states), (attentions)

    
    
 
