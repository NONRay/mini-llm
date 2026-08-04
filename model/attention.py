import torch
import torch.nn as nn
import torch.nn.functional as F



class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.n_head=config.n_head
        self.hidden_dim=config.hidden_dim


        self.qkv = nn.Linear(
            config.hidden_dim,
            config.hidden_dim*3
        )


        self.proj = nn.Linear(
            config.hidden_dim,
            config.hidden_dim
        )


        self.dropout=nn.Dropout(
            config.dropout
        )


        mask=torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len
            )
        )


        self.register_buffer(
            "mask",
            mask.view(
                1,1,
                config.max_seq_len,
                config.max_seq_len
            )
        )



    def forward(
        self,
        x,
        return_attention=False
    ):

        B,T,C=x.shape


        qkv=self.qkv(x)


        q,k,v=qkv.chunk(3,dim=-1)


        q=q.view(
            B,T,
            self.n_head,
            C//self.n_head
        ).transpose(1,2)


        k=k.view(
            B,T,
            self.n_head,
            C//self.n_head
        ).transpose(1,2)


        v=v.view(
            B,T,
            self.n_head,
            C//self.n_head
        ).transpose(1,2)



        att=(q@k.transpose(-2,-1))


        att=att/(k.size(-1)**0.5)


        att=att.masked_fill(
            self.mask[:,:,:T,:T]==0,
            float("-inf")
        )


        att=F.softmax(
            att,
            dim=-1
        )


        att=self.dropout(att)



        y=att@v


        y=y.transpose(
            1,2
        ).contiguous().view(
            B,T,C
        )


        y=self.proj(y)



        if return_attention:

            return y,att


        return y
