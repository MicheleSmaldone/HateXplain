# import torch
# import transformers
# #from keras.preprocessing.sequence import pad_sequences
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
    
   
#     encoder = LabelEncoder()
    
#     encoder.classes_ = np.load(params['class_names'],allow_pickle=True)
#     labels=encoder.transform(labels)
    
#     input_ids = pad_sequences(input_ids,maxlen=int(params['max_length']), dtype="long", 
#                           value=0, truncating="post", padding="post")
#     att_vals = pad_sequences(att_vals,maxlen=int(params['max_length']), dtype="float", 
#                           value=0.0, truncating="post", padding="post")
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
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder

def pad_sequences(batch, max_len, padding_value=0):
    """
    batch: List[List[int]]  (each inner list is a token-ID sequence)
    max_len: int            (the fixed length you want every example to be)
    """
    out = []
    for seq in batch:
        # truncate
        seq_t = torch.tensor(seq, dtype=torch.long)[:max_len]
        # pad on the right if needed
        if seq_t.size(0) < max_len:
            pad_amt = max_len - seq_t.size(0)
            seq_t = F.pad(seq_t, (0, pad_amt), value=padding_value)
        out.append(seq_t)
    # stack into [batch_size, max_len]
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

    # encode labels
    encoder = LabelEncoder()
    encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
    labels = encoder.transform(labels_list)

    # pad everything to exactly max_length
    max_len = int(params['max_length'])
    input_ids = pad_sequences(input_ids_list, max_len, padding_value=0)
    att_vals  = pad_sequences(att_vals_list,  max_len, padding_value=0.0)

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
