import torch
import torch.nn as nn

from .llama_block import TransformerBlock,RMSNorm



class MiniLLM(nn.Module):


    def __init__(self,config):

        super().__init__()


        self.token_embedding=nn.Embedding(
            config.vocab_size,
            config.hidden_dim
        )


        self.position_embedding=nn.Embedding(
            config.max_seq_len,
            config.hidden_dim
        )


        self.blocks=nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )


        self.norm=RMSNorm(
            config.hidden_dim
        )


        self.lm_head=nn.Linear(
            config.hidden_dim,
            config.vocab_size,
            bias=False
        )


        self.lm_head.weight=self.token_embedding.weight



    def forward(
        self,
        idx,
        targets=None,
        return_hidden=False
    ):


        B,T=idx.shape


        pos=torch.arange(
            T,
            device=idx.device
        )


        x=(
            self.token_embedding(idx)
            +
            self.position_embedding(pos)
        )


        hidden=[]


        for block in self.blocks:

            x=block(x)

            if return_hidden:

                hidden.append(x)



        x=self.norm(x)


        logits=self.lm_head(x)



        loss=None

        if targets is not None:

            loss=nn.functional.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1)
            )



        if return_hidden:

            return logits,loss,hidden


        return logits,loss
