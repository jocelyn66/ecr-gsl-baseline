import torch
import torch.nn as nn
from utils.name2object import name2gsl, name2init
import models.gsl as gsls
import tqdm
import numpy as np
from utils.train import normalize_adjacency, preprocess_adjacency


class ECRModel(nn.Module):

    def __init__(self, args, tokenizer, plm_model, schema_list, orig_adj=None):
        super(ECRModel, self).__init__()

        # bert
        self.tokenizer = tokenizer
        self.bert_encoder = plm_model
        self.doc_schema = schema_list[0]
        self.event_schema = schema_list[1]
        self.entity_schema = schema_list[2]
        self.win_w = args.win_w
        self.win_len = args.win_len
        self.concat_cls = args.cls

        self.gsl = getattr(gsls, name2gsl[args.encoder])(args.feat_dim, args.hidden1, args.hidden2, args.dropout)
        self.gsl_name = args.encoder
        self.device = args.device

        # regularization
        self.loss_type = args.loss_type
        if self.loss_type in [2,3,4]:

            # init_adj = orig_adj / 2
            init_adj = self.eye_init(args.n_nodes['Train'])
            # init_adj = self.rand_init(args.n_nodes['Train'])

            init_adj = preprocess_adjacency(init_adj)
            self.W = nn.Parameter(torch.tensor(init_adj), requires_grad=True)
            # # self.W = nn.Parameter(torch.tensor(orig_adj / 2), requires_grad=True)
            # self.W = nn.Parameter(torch.tensor(self.eye_init(args.n_nodes['Train'], args.double_precision)), requires_grad=True)
            # # self.W = nn.Parameter(torch.tensor(self.rand_init(args.n_nodes['Train'])), requires_grad=True)

        if self.loss_type in [2, 4]:

            # init_adj = orig_adj / 2
            init_adj = self.eye_init(args.n_nodes['Train'])
            # init_adj = self.rand_init(args.n_nodes['Train'])

            init_adj = preprocess_adjacency(init_adj)
            self.H = nn.Parameter(torch.tensor(init_adj), requires_grad=True)

    def rand_init(self, n_nodes, rand_rate=0.1):
        # 随机+对称
        mat = np.random.rand(n_nodes, n_nodes)
        mat = np.where(mat<rand_rate, 1, 0)
        mat = mat + np.eye(n_nodes)
        #保证度>0
        # rows = np.arange(n_nodes)
        # cols = np.arange(n_nodes)
        # np.random.shuffle(cols)
        # mat[rows, cols] = 1
        # return np.where(mat + mat.T>0, 1.,0.)
        return (mat + mat.T)/2

    def eye_init(self, n_nodes):
        # if double_precision:
        #     return np.eye(n_nodes)
        # else:
        #     return np.eye(n_nodes).astype(np.float32)
        return np.eye(n_nodes)

    def forward(self, dataset, adj):
        # 待优化
        features = []  # ts list

        # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32)

        # 遍历句子构造句子子图, 同时记录句子文档id
        for _, sent in enumerate(dataset):

            # batch_size, seq_len = sent['input_ids'].shape

            input_ids = torch.tensor(sent['input_ids'], device=self.device).reshape(1, -1)
            
            encoder_output = self.bert_encoder(input_ids)
            encoder_hidden_state = torch.mm(torch.tensor(sent['input_mask'], device=self.device).T, encoder_output['last_hidden_state'].squeeze())  # (n_sent, n_tokens, feat_dim)

            masks = []
            # token_masks = torch.tensor(sent['input_mask'])
            word_masks = torch.eye(encoder_hidden_state.shape[0], device=self.device)[1:-1]

            for _, event in enumerate(sent['output_event']):

                mention_mask = torch.zeros((1, encoder_hidden_state.shape[0]-2), device=self.device)  #不含[CLS][SEP]的索引
                mention_mask[:, event['tokens_number']] = 1  #'token_number':起始0 不考虑cls, encoder_hidden_state 需要+1

                if self.win_len>0 and self.win_w>0:
                    mention_mask = self.expand_win(event['tokens_number'], mention_mask)

                mask_sum = mention_mask.sum(dim=1)
                mask_ave = (1 / mask_sum).repeat((1, mention_mask.shape[1]))
                mention_mask = mention_mask * mask_ave

                this_mask = torch.mm(mention_mask, word_masks)  #mention mask(考虑了cls)
                masks.append(this_mask)

            for _,entity in enumerate(sent['output_entity']):

                mention_mask = torch.zeros((1, encoder_hidden_state.shape[0]-2), device=self.device)  #不含[CLS][SEP]的索引
                mention_mask[:, entity['tokens_number']] = 1

                if self.win_len>0 and self.win_w>0:
                    mention_mask = self.expand_win(entity['tokens_number'], mention_mask)

                mask_sum = mention_mask.sum(dim=1)
                mask_ave = (1 / mask_sum).repeat((1, mention_mask.shape[1]))
                mention_mask = mention_mask * mask_ave

                this_mask = torch.mm(mention_mask, word_masks)

                masks.append(this_mask)
                
            masks = torch.cat(masks, dim=0).cuda()
            this_feature = torch.mm(masks, encoder_hidden_state)
            if self.concat_cls:
                this_feature = torch.concat((this_feature, encoder_hidden_state[0:1, :].repeat((this_feature.shape[0], 1))), dim=1)            
            features.append(this_feature)

        features = torch.cat(features)    # encoder_hidden_state * input_mask = 所求表征
        return self.gsl(features, adj)  # gae
    
    def expand_win(self, numbers, mask):
        max_count, count, l, max_l, max_r = 0, 0, numbers[0], numbers[0], numbers[-1]
        nums = [numbers[0]-1]+numbers+[numbers[-1]+2]
        for i in range(1, len(nums)-1):
            
            if nums[i] - nums[i-1] > 1:  #不连续
                if count>max_count:
                    max_count = count
                    max_l = l
                    max_r = nums[i-1]
                l = nums[i]
                count=1
            else:
                count += 1
        expand = mask.clone()
        expand[:, max(max_l-self.win_len, 0):max_l] = self.win_w
        expand[:, max_r+1:min(max_r+1+self.win_len, mask.shape[-1])] = self.win_w
        return torch.where(mask>0, mask, expand)
