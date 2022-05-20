import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class FMLayer(nn.Module):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_number) -> None:
        super().__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_number = dense_feature_number
        self.fm = FM(sparse_feature_number, sparse_feature_dim,
                 dense_feature_number)

        # zero order
        self.zero_bias = nn.Parameter(
            torch.zeros((1))
        ) 

    def weight_init(self):
        self.fm.weight_init()

    def forward(self, sparse_inputs, dense_inputs):
        score_first_order, score_second_order = self.fm.forward(sparse_inputs, dense_inputs)
        predict = torch.sigmoid(score_first_order + score_second_order + self.zero_bias)
        return predict



class FM(nn.Module):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_number) -> None:
        super().__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_number = dense_feature_number
        self.dense_feature_dim = sparse_feature_dim

        # sparse first order
        self.sparse_embedding_one = nn.Embedding(
            num_embeddings=sparse_feature_number,
            embedding_dim=1,
            sparse = True
        )

        # sparse second order
        self.sparse_embedding_second = nn.Embedding(
            num_embeddings = sparse_feature_number,
            embedding_dim = sparse_feature_dim,
            sparse = True
        )

        # dense first order
        self.dense_embedding_one = nn.Parameter(
            torch.zeros((dense_feature_number))
        )

        # dense second order
        self.dense_embedding_second = nn.Parameter(
            torch.zeros((1, dense_feature_number, self.dense_feature_dim))
        )        
    
    def weight_init(self):
        nn.init.normal_(self.sparse_embedding_one.weight.data, mean=0.0,std=0.1 / math.sqrt(float(self.sparse_feature_dim)))
        nn.init.normal_(self.sparse_embedding_second.weight.data, mean=0.0,std=0.1 / math.sqrt(float(self.sparse_feature_dim)))
        nn.init.constant_(self.dense_embedding_one, val=1.0)
        nn.init.constant_(self.dense_embedding_second, val=1.0)

    def forward(self, sparse_inputs, dense_inputs):
        # first order 
        sparse_emb_one = self.sparse_embedding_one(sparse_inputs)
        dense_emb_one = self.dense_embedding_one.mul((dense_inputs))
        dense_emb_one = torch.unsqueeze(dense_emb_one, dim=2)
        score_first_order = torch.sum(sparse_emb_one, axis=1) + torch.sum(dense_emb_one, axis=1)

        #second order
        sparse_embedding = self.sparse_embedding_second(sparse_inputs)
        dense_inputs_re = torch.unsqueeze(dense_inputs, dim = 2)
        dense_embedding = self.dense_embedding_second.mul(dense_inputs_re)
        fea_embedding = torch.concat([sparse_embedding, dense_embedding], axis=1)

        ## sum square part
        summed_feature_embedding = torch.sum(fea_embedding, axis=1)
        summed_feature_embedding_square = torch.square(summed_feature_embedding)
        ## square sum part
        square_feature_embedding = torch.square(fea_embedding)
        square_sum_feature_embedding = torch.sum(square_feature_embedding, axis=1)

        score_second_order = 0.5 * torch.sum(
            summed_feature_embedding_square - square_sum_feature_embedding, 
            axis=1, 
            keepdim=True)

        return score_first_order, score_second_order


if __name__ == "__main__":
    sparse_feature_number = 10
    sparse_feature_dim = 3
    dense_feature_number = 2

    # a = FM(sparse_feature_number, sparse_feature_dim, dense_feature_number)

    # two case: sparse [1,2,3], dense [1.1,2,2]  sparse [4,5,6], dense [4.4,5,5]  
    sparse_inputs = [torch.LongTensor([[1],[4]]), torch.LongTensor([[2],[5]]), torch.LongTensor([[3],[6]])] 
    sparse_inputs = torch.concat(sparse_inputs, axis=1)
    dense_inputs = torch.Tensor([[1.1, 2.2], [4.4, 5.5]])

    # res = a.forward(sparse_inputs, dense_inputs)

    b = FMLayer(sparse_feature_number, sparse_feature_dim, dense_feature_number)
    predict = b(sparse_inputs, dense_inputs)    
    print(predict)
