# import torch
# import transformers
# from keras.preprocessing.sequence import pad_sequences
# from torch.nn.utils.rnn import pad_sequence

# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# import numpy as np
# from sklearn.preprocessing import LabelEncoder


# def pad_sequences(batch, max_len, padding_value=0):
#     # batch: list[list[int]]
#     batch = [torch.tensor(x[:max_len]) for x in batch]
#     return pad_sequence(batch, batch_first=True, padding_value=padding_value)

# def custom_att_masks(input_ids):
#     attention_masks = []

#     # For each sentence...
#     for sent in input_ids:

#         # Create the attention mask.
#         #   - If a token ID is 0, then it's padding, set the mask to 0.
#         #   - If a token ID is > 0, then it's a real token, set the mask to 1.
#         att_mask = [int(token_id > 0) for token_id in sent]

#         # Store the attention mask for this sentence.
#         attention_masks.append(att_mask)
#     return attention_masks

# def combine_features(tuple_data,params,is_train=False):
#     input_ids =  [ele[0] for ele in tuple_data]
#     att_vals = [ele[1] for ele in tuple_data]
#     labels = [ele [2] for ele in tuple_data]
    
   
#     # --- DEBUG BLOCK: check raw attention vectors before padding ---
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> combine_features DEBUG:")
#     print("  sample raw att_vals_list[0][:10]:", att_vals[0][:10])
#     print("  sum raw att_vals_list[0]:", sum(att_vals[0]))
#     print("  dtype raw att_vals_list:", type(att_vals[0][0]))
#     nonzero_counts = [sum(1 for x in v if x!=0) for v in att_vals[:5]]
#     print("  nonzero count in first 5 samples:", nonzero_counts)
#     # ------------------------------------------------------------------
#     encoder = LabelEncoder()
    
#     encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
#     labels=encoder.transform(labels)
    
#     input_ids = pad_sequences(input_ids,maxlen=int(params['max_length']), dtype="long", 
#                           value=0, truncating="post", padding="post")
#     att_vals = pad_sequences(att_vals,maxlen=int(params['max_length']), dtype="float", 
#                           value=0.0, truncating="post", padding="post")
    
#     # --- DEBUG BLOCK: after padding ---
#     print("  after padding att_vals[0][:10]:", att_vals[0][:10].tolist())
#     print("  sum padded att_vals[0]:", att_vals[0].sum().item())
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> combine_features DEBUG:")

#     # ------------------------------------------------------------------

#     att_masks=custom_att_masks(input_ids)
#     dataloader=return_dataloader(input_ids,labels,att_vals,att_masks,params,is_train)
#     return dataloader

# def return_dataloader(input_ids,labels,att_vals,att_masks,params,is_train=False):
#     inputs = torch.tensor(input_ids)
#     labels = torch.tensor(labels,dtype=torch.long)
#     masks = torch.tensor(np.array(att_masks),dtype=torch.uint8)
#     attention = torch.tensor(np.array(att_vals),dtype=torch.float)
#     data = TensorDataset(inputs,attention,masks,labels)
#     if(is_train==False):
#         sampler = SequentialSampler(data)
#     else:
#         sampler = RandomSampler(data)
#     dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
#     return dataloader

# ------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch.nn.functional as F

def pad_sequences(seqs, max_len, padding_value=0, dtype=torch.long):
    out = []
    for seq in seqs:
        t = torch.tensor(seq, dtype=dtype)[:max_len]
        if t.size(0) < max_len:
            pad_amt = max_len - t.size(0)
            t = F.pad(t, (0, pad_amt), value=padding_value)
        out.append(t)
    return torch.stack(out, dim=0)


def custom_att_masks(input_ids_tensor):
    """
    input_ids_tensor: torch.LongTensor of shape [batch_size, seq_len]
    Returns a list-of-lists mask where token_id>0 → 1, else 0.
    """
    masks = []
    for seq in input_ids_tensor.tolist():
        masks.append([int(tok > 0) for tok in seq])
    return masks

def combine_features(tuple_data, params, is_train=False):
    # unpack the raw tuples
    input_ids_list = [ele[0] for ele in tuple_data]
    att_vals_list  = [ele[1] for ele in tuple_data]
    labels_list    = [ele[2] for ele in tuple_data]


    # --- DEBUG BLOCK: check raw attention vectors before padding ---
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> combine_features DEBUG:")
    # print("  sample raw att_vals_list[0][:10]:", att_vals_list[0][:10])
    # print("  sum raw att_vals_list[0]:", sum(att_vals_list[0]))
    # print("  dtype raw att_vals_list:", type(att_vals_list[0][0]))
    # nonzero_counts = [sum(1 for x in v if x!=0) for v in att_vals_list[:5]]
    # print("  nonzero count in first 5 samples:", nonzero_counts)
    # ------------------------------------------------------------------

    # encode labels
    encoder = LabelEncoder()
    encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
    labels = encoder.transform(labels_list)

    # pad everything to exactly max_length
    max_len = int(params['max_length'])
    # input_ids = pad_sequences(input_ids_list, max_len, padding_value=0)
    # att_vals  = pad_sequences(att_vals_list,  max_len, padding_value=0.0)

    # pad token IDs
    input_ids = pad_sequences(input_ids_list,
                              max_len,
                              padding_value=0,
                              dtype=torch.long)
    # pad attention values (must use float dtype!)
    att_vals  = pad_sequences(att_vals_list,
                              max_len,
                              padding_value=0.0,
                              dtype=torch.float)

    # --- DEBUG BLOCK: after padding ---
    # print("  after padding att_vals[0][:10]:", att_vals[0][:10].tolist())
    # print("  sum padded att_vals[0]:", att_vals[0].sum().item())
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> combine_features DEBUG:")

    # ------------------------------------------------------------------

    # build attention masks from the padded input_ids
    att_masks = custom_att_masks(input_ids)

    return return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train)

def return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train=False):
    # convert into tensors
    inputs    = input_ids               # already a torch.LongTensor
    attention = att_vals                # already a torch.LongTensor/FloatTensor
    masks     = torch.tensor(att_masks, dtype=torch.uint8)
    labels    = torch.tensor(labels,    dtype=torch.long)

    dataset = TensorDataset(inputs, attention, masks, labels)
    sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=params['batch_size'])
