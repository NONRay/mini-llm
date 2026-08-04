import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention



class RMSNorm(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.weight=nn.Parameter(
            torch.ones(dim)
        )

        self.eps=1e-6



    def forward(self,x):

        norm=x.pow(2).mean(-1,keepdim=True)

        x=x*torch.rsqrt(
            norm+self.eps
        )

        return self.weight*x




class SwiGLU(nn.Module):

    def __init__(self,dim):

        super().__init__()

        hidden=dim*4


        self.w1=nn.Linear(
            dim,
            hidden
        )

        self.w2=nn.Linear(
            hidden,
            dim
        )

        self.w3=nn.Linear(
            dim,
            hidden
        )



    def forward(self,x):

        return self.w2(
            F.silu(
                self.w1(x)
            )
            *
            self.w3(x)
        )




class TransformerBlock(nn.Module):

    def __init__(self,config):

        super().__init__()

        self.norm1=RMSNorm(
            config.hidden_dim
        )

        self.attn=CausalSelfAttention(
            config
        )


        self.norm2=RMSNorm(
            config.hidden_dim
        )


        self.ffn=SwiGLU(
            config.hidden_dim
        )



    def forward(
        self,
        x,
        return_attention=False
    ):


        if return_attention:

            h,att=self.attn(
                self.norm1(x),
                True
            )

            x=x+h

        else:

            x=x+self.attn(
                self.norm1(x)
            )


        x=x+self.ffn(
            self.norm2(x)
        )


        if return_attention:

            return x,att


        return x
