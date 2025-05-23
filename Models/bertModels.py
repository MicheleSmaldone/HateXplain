import torch
import math
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
#from .utils import masked_cross_entropy  # still available if needed elsewhere


def sparsemax_tensor(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax that works on shape (B, H, L) or (B, L).
    Returns a tensor of the same shape, with gradients.
    """
    # 1) sort z in descending order along dim
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)

    # 2) compute cumulative sums of sorted z
    cssv = torch.cumsum(z_sorted, dim=dim)

    # 3) create a 1..K vector for threshold computation
    k = z.size(dim)
    arange = torch.arange(1, k + 1, device=z.device).view(
        *([1] * (z.dim() - 1)), k
    )

    # 4) determine sparsity support: where 1 + k * z_sorted > cssv
    support = 1 + arange * z_sorted > cssv

    # 5) find the largest k in support for each slice
    k_max = support.sum(dim=dim, keepdim=True)

    # 6) compute threshold τ = (sum_{j ≤ k_max} z_sorted_j - 1) / k_max
    tau = (cssv.gather(dim, k_max - 1) - 1) / k_max

    # 7) compute output = max(z - τ, 0)
    return torch.clamp(z - tau, min=0)


# def sparsemax_onehot_loss(
#     z: torch.Tensor,
#     k: torch.LongTensor,
#     mask: torch.Tensor = None,
#     reduction: str = "mean"
# ) -> torch.Tensor:
#     """
#     Implements Eq. (3.3) for one-hot targets k:
#       L(z; k) = - z_k + ½ ∑_{j ∈ S(z)} [ z_j² - τ(z)² ] + ½

#     Args:
#       z      : raw logits, shape (B, L)
#       k      : ground-truth indices, shape (B,) with 0 <= k[b] < L
#       mask   : optional 0/1 mask, shape (B, L) to ignore padding
#       reduction: "mean" | "sum" | "none"
#     Returns:
#       loss   : scalar (if reduced) or shape (B,) if reduction="none"
#     """
#     B, L = z.size()

#     # 1) sparsemax and support S(z)
#     p = sparsemax_tensor(z, dim=-1)
#     support = p > 0

#     # 2) gather z_k
#     z_k = z.gather(1, k.unsqueeze(1)).squeeze(1)

#     # 3) threshold τ = (sum_{j∈S} z_j - 1) / |S|
#     sum_z  = (z * support).sum(dim=1)
#     S_size = support.sum(dim=1).float()
#     tau    = (sum_z - 1.0) / S_size
#     print("---------------- S_size: ", S_size)
#     # <<< DEBUG BLOCK >>>
#     zero_support = (S_size == 0).nonzero(as_tuple=False)
#     if zero_support.numel() > 0:
#         print(f"⚠️  Empty sparsemax support on examples {zero_support.flatten().tolist()}")
#         # you can also peek at the raw scores that caused it:
#         bad_idxs = zero_support.flatten()
#         print(" raw logits for these examples:", z[bad_idxs].detach().cpu().numpy())
#         # and their target positions:
#         print(" targets:", k[bad_idxs].detach().cpu().numpy())
#     # <<< END DEBUG >>>

#     # 4) quadratic term ½ ∑_{j∈S} (z_j² - τ²)
#     sum_z2 = ((z * z) * support).sum(dim=1)
#     term   = 0.5 * (sum_z2 - S_size * tau**2)

#     # 5) assemble loss
#     loss = -z_k + term + 0.5

#     # 6) mask out padded examples
#     if mask is not None:
#         valid = (mask.sum(dim=1) > 0).float()
#         loss  = loss * valid

#     # 7) reduction
#     if reduction == "mean":
#         return loss.sum() / valid.sum() if mask is not None else loss.mean()
#     if reduction == "sum":
#         return loss.sum()
#     return loss  # none

def sparsemax_onehot_loss(z, k, mask=None, reduction="mean"):
    p       = sparsemax_tensor(z, dim=-1)
    support = p > 0
    S_size  = support.sum(dim=1).float()              # (B,)

    # clamp S_size to avoid div-by-0
    safe_S  = torch.clamp(S_size, min=1.)
    tau     = ( (z * support).sum(dim=1) - 1.0 ) / safe_S

    quad    = 0.5 * ( ((z*z) * support).sum(dim=1) - safe_S * tau**2 )
    z_k     = z.gather(1, k.unsqueeze(1)).squeeze(1)
    loss    = -z_k + quad + 0.5

    # mask padded examples if supplied
    if mask is not None:
        valid = (mask.sum(dim=1) > 0).float()
        loss  = loss * valid
        denom = valid.sum()
    else:
        denom = z.size(0)

    if reduction == "mean": return loss.sum() / denom
    if reduction == "sum" : return loss.sum()
    return loss

class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.num_labels   = config.num_labels
        self.weights      = params['weights']
        self.train_att    = params['train_att']
        self.lam          = params['att_lambda']
        self.num_sv_heads = params['num_supervised_heads']
        self.sv_layer     = params['supervised_layer_pos']

        # existing BERT + classifier
        self.bert       = BertModel(config)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        # hook buffer
        self._raw_attn = {}

        # select layers from params or fallback to single sv_layer
        supervise_layers = params.get('supervise_layers', [self.sv_layer])

        def make_raw_cls_hook(layer_idx):
            # def hook(module, inputs, outputs):
            #     hidden_states, attn_mask = inputs[0], inputs[1]
            #     q = module.query(hidden_states)
            #     k = module.key(hidden_states)
            #     bs, seq_len, _ = hidden_states.size()
            #     q = module.transpose_for_scores(q)
            #     k = module.transpose_for_scores(k)
            #     # scores = torch.matmul(q, k.transpose(-1, -2))
            #     # scores = scores / math.sqrt(module.attention_head_size)
            #     scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(...)
            #     # attn_mask has -1e4 (or 0) for each position:
            #     scores = scores.masked_fill(attn_mask < -1e3, -1e4)
            #     self._raw_attn[layer_idx] = scores[:, :, 0, :]
            #     if attn_mask is not None:
            #         scores = scores + attn_mask
            #     # store CLS row for all heads
            #     self._raw_attn[layer_idx] = scores[:, :, 0, :]
            # return hook
            def hook(module, inputs, outputs):
                hidden_states, attn_mask = inputs[0], inputs[1]
                # project to queries & keys
                q = module.query(hidden_states)
                k = module.key(hidden_states)
                # reshape for multi-head
                q = module.transpose_for_scores(q)
                k = module.transpose_for_scores(k)
                # compute raw dot-product scores
                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores / math.sqrt(module.attention_head_size)
                # now mask out the padded positions to a reasonable floor
                # attn_mask is 0 for real tokens, large negative for pads
                # so we invert the test: wherever attn_mask is "really negative", we force −1e4
                scores = scores.masked_fill(attn_mask < -1e3, -1e4)
                # finally, grab the CLS→all-tokens slice for this layer
                self._raw_attn[layer_idx] = scores[:, :, 0, :]
            return hook

        for L in supervise_layers:
            self.bert.encoder.layer[L].attention.self.register_forward_hook(
                make_raw_cls_hook(L)
            )

    # ============== #
    # SPARSEMAX + CROSS
    # ============== #

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     attention_vals=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     labels=None,
    #     device=None
    # ):
    #     outputs = self.bert(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=True
    #     )

    #     # classification head
    #     pooled_output = outputs[1]
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     outputs = (logits,) + outputs[2:]

    #     if labels is not None:
    #         if torch.isnan(loss):
    #             print("⚠️  NaN detected in total loss")
    #             print("  - CE loss:", loss_f.item())
    #             print("  - λ:", self.lam)
    #         loss_f = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
    #         if torch.isnan(loss_f):
    #             print("⚠️  NaN in classification loss – check your labels/logits")
    #         loss = loss_f(logits.view(-1, self.num_labels), labels.view(-1))

    #         if self.train_att:
    #             loss_att = 0.0
    #             attention_idx = attention_vals.argmax(dim=1)
    #             for h in range(self.num_sv_heads):
    #                 cls_raw = self._raw_attn[self.sv_layer][:, h, :]
    #                 loss_att += self.lam * sparsemax_onehot_loss(
    #                     cls_raw,
    #                     attention_idx,
    #                     mask=attention_mask
    #                 )
    #             # change self.lam * sparsemax_loss because for now it's a number TODO
    #             loss = loss + loss_att

    #             if torch.isnan(loss_att):
    #                 print("⚠️  NaN in attention‐supervision loss (head %d)" % h)
    #                 bad = torch.isnan(cls_raw).any(dim=1) | torch.isinf(cls_raw).any(dim=1)
    #                 print("⇢ bad examples:", bad.nonzero(as_tuple=False).squeeze(-1).tolist())
    #                 print("⇢ CLS raw min/max:",
    #                     cls_raw.min().item(), cls_raw.max().item())
    #                 # print("⇢ S_size zeros:",
    #                 #     (support.sum(dim=1) == 0).nonzero(as_tuple=False).squeeze(-1).tolist())


    #         outputs = (loss,) + outputs

    #     return outputs  # (loss), logits, hidden_states, attentions

    # ============== #
    # SPARSEMAX ONLY
    # ============== #

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     attention_vals=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     labels=None,
    #     device=None
    # ):
    #     # 1) run BERT
    #     outputs = self.bert(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=True
    #     )
    #     pooled_output = outputs[1]

    #     # 2) classification head
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #     outputs = (logits,) + outputs[2:]  # keep hidden_states, attentions

    #     if labels is not None:
    #         # —— classification loss via sparsemax——
    #         cls_loss = sparsemax_onehot_loss(
    #             logits.view(-1, self.num_labels),
    #             labels.view(-1),
    #             mask=None,
    #             reduction="mean"
    #         )
    #         if torch.isnan(cls_loss):
    #             print("⚠️  NaN in classification (sparsemax) loss – check logits/labels")

    #         # —— attention‐supervision loss ——  
    #         if self.train_att:
    #             loss_att = 0.0
    #             # attention_vals: (B, L) your target positions
    #             attention_idx = attention_vals.argmax(dim=1)
    #             for h in range(self.num_sv_heads):
    #                 cls_raw = self._raw_attn[self.sv_layer][:, h, :]  # (B, L)
    #                 with torch.no_grad():                         # no gradient needed
    #                     p       = sparsemax_tensor(cls_raw, dim=-1)
    #                     support = p > 0                           # (B, L) boolean
    #                     zero_sup = (support.sum(dim=1) == 0)      # (B,)
    #                     if zero_sup.any():
    #                         bad_idx = zero_sup.nonzero(as_tuple=False).squeeze(-1)
    #                         print(f"⚠️  empty sparsemax support in batch, head {h}:",
    #                             bad_idx.tolist())
    #                         print("   raw min/max:",
    #                             cls_raw.min().item(), cls_raw.max().item())
    #                         print("   sample logits of first bad ex:",
    #                             cls_raw[bad_idx[0]].cpu().tolist()[:20])
    #                 loss_att += self.lam * sparsemax_onehot_loss(
    #                     cls_raw,
    #                     attention_idx,
    #                     mask=attention_mask,
    #                     reduction="mean"
    #                 )
    #             if torch.isnan(loss_att):
    #                 print("⚠️  NaN in attention‐supervision loss (head %d)" % h)
    #                 bad = torch.isnan(cls_raw).any(dim=1) | torch.isinf(cls_raw).any(dim=1)
    #                 print("⇢ bad examples:", bad.nonzero(as_tuple=False).squeeze(-1).tolist())
    #                 print("⇢ CLS raw min/max:",
    #                     cls_raw.min().item(), cls_raw.max().item())
    #                 # print("⇢ S_size zeros:",
    #                 #     (support.sum(dim=1) == 0).nonzero(as_tuple=False).squeeze(-1).tolist())

    #             loss = cls_loss + loss_att
    #         else:
    #             loss = cls_loss

    #         if torch.isnan(loss):
    #             print("⚠️  NaN detected in total loss:",
    #                 " cls_loss=", cls_loss.item(),
    #                 " loss_att=",
    #                 (loss_att.item() if self.train_att else 0.0),
    #                 " λ=", self.lam)

    #         outputs = (loss,) + outputs

    #     return outputs  # (loss), logits, hidden_states, attentions

    # ============== #
    # SPARSEMAX + CROSS
    # ============== #

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None
    ):
        # 1) run BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True
        )
        pooled_output = outputs[1]

        # 2) classification head
        pooled_output = self.dropout(pooled_output)
        logits       = self.classifier(pooled_output)
        outputs      = (logits,) + outputs[2:]  # logits, hidden_states, attentions...

        if labels is not None:
            # —— 1) standard CrossEntropyLoss on the logits ——  
            ce = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            cls_loss = ce(logits.view(-1, self.num_labels),
                        labels.view(-1))
            
            # —— 2) sparsemax‐based loss *only* on the attention heads ——  
            loss_att = 0.0
            if self.train_att:
                attention_idx = attention_vals.argmax(dim=1)
                for h in range(self.num_sv_heads):
                    cls_raw = self._raw_attn[self.sv_layer][:, h, :]  # (B, L)
                    loss_att += self.lam * sparsemax_onehot_loss(
                        cls_raw,
                        attention_idx,
                        mask=attention_mask,
                        reduction="mean"
                    )

            # combine them:
            loss = cls_loss + loss_att

            # sanity‐checks
            if torch.isnan(cls_loss):
                print("⚠️  NaN in CE loss")
            if torch.isnan(loss_att):
                print("⚠️  NaN in sparsemax attention loss")
            if torch.isnan(loss):
                print("⚠️  Total loss is NaN (ce, att) =", cls_loss.item(), loss_att.item())

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, hidden_states, attentions
