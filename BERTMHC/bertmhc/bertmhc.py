# coding=utf-8
# NAME OF THE PROGRAM THIS FILE BELONGS TO
#
#       BERTMHC
#
#   file: cli.py
#
#    Authors: Jun Cheng (jun.cheng@neclab.eu)
#             Brandon Malone (brandon.malone@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2020, All rights reserved.
#     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#     PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
#
#                                                                                                                                                                           This is a license agreement ("Agreement") between your academic institution or non-profit organization or self (called "Licensee" or "You" in this Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF LICENSOR’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials and to use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication. All publication materials mentioning features or use of this software must explicitly include an acknowledgement the software was developed by NEC Laboratories Europe GmbH.
# COPYRIGHT: The Software is owned by Licensor.
#     PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
# DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
# BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement. Neither the name of NEC Laboratories Europe GmbH nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
# TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
# The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
# FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
#     DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement.
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs administered by law.
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Germany without reference to conflict of laws principles.  You consent to the personal jurisdiction of the courts of this country and waive their rights to venue outside of Germany.
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.



"""Main module."""
from tape import ProteinBertAbstractModel, ProteinBertModel
from tape.models.modeling_utils import SimpleMLP
import torch
import torch.nn as nn
import torch.nn.functional as F

''' # MHCHead
(0): Linear(in_features=768, out_features=512, bias=True)
(1): ReLU()
(2): Dropout(p=0.0, inplace=True)
(3): Linear(in_features=512, out_features=2, bias=True)
'''
class MHCHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 512, num_labels)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        #print(logits.shape) #(batch_size, 2)
        outputs = (logits, )
        if targets is not None:            
            outputs = logits, targets   
        return outputs  # logits, (targets)

class BERTMHC(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        self.classify = MHCHead(
            config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output = outputs[:2]        
        average = torch.mean(sequence_output, dim=1)
        logits = self.classify(average, targets)
        outputs = logits + outputs[2:]  
        return outputs

# BERTMHC linnear, BERT con una capa lineal al final
class BERTMHC_LINEAR(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)       

        self.bert       = ProteinBertModel(config)
        self.dropout    = nn.Dropout(0.0)
        self.relu       = nn.ReLU()
        self.linear_1   = nn.Linear(config.hidden_size, 512)
        self.linear_2   = nn.Linear(512, 2)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)        

        sequence_output, pooled_output = outputs[:2]  
        average = torch.mean(sequence_output, dim=1)

        out     = self.linear_1(average)
        out     = self.relu(out)
        out     = self.dropout(out)
        logits  = self.linear_2(out)

        out = (logits, )
        if targets is not None:
            out = logits, targets
        outputs = out + outputs[2:]
        return outputs
    
# RNN with Attention
class BERTMHC_RNN_ATT(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)       

        self.bert       = ProteinBertModel(config)

        self.num_labels = config.num_labels        
        self.hidden_size = config.hidden_size
        self.num_rnn_layer = 2
        self.rnn_dropout = 0.1
        self.rnn_hidden = 768
        self.max_seq_len = 51
        self.att_dropout = 0.1

        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.rnn_hidden, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)        

        self.w_omega = nn.Parameter(torch.Tensor(self.rnn_hidden * 2, self.rnn_hidden * 2))
        self.u_omega = nn.Parameter(torch.Tensor(self.rnn_hidden * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.dropout = nn.Dropout(self.rnn_dropout)
        self.att_dropout = nn.Dropout(self.att_dropout)
        self.classifier = nn.Linear(2*self.rnn_hidden, config.num_labels)

        self.init_weights()

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) /math.sqrt(d_k)  #   scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)
        rattention = torch.matmul(p_attn, x)
        context = torch.matmul(p_attn, x).sum(1)  # [batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, rattention

    def forward(self, input_ids, input_mask=None, targets=None):
        outputs = self.bert(input_ids, input_mask=input_mask)        

        sequence_output, pooled_output = outputs[:2]  
        average = torch.mean(sequence_output, dim=1)

        # lstm
        rnn_out, (ht, ct) = self.rnn(pooled_output)
        query = self.att_dropout(rnn_out)
        attn_output, attention = self.attention_net(rnn_out, query)  # 
        logits = self.classifier(attn_output)

        out = (logits, )
        if targets is not None:
            out = logits, targets
        outputs = out + outputs[2:]
        return outputs

# ACTUALIZACIÓN, UTILZIANDO UNA 1DCNN
""" # it owrks poorly
class BERTMHC_CNN(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)

        self.layer1 = nn.Sequential(
            nn.Conv1d(60, 128, kernel_size=3), # 60 xq tenemos 60 aminoacidos
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(9728,4000),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(4000,600),
            nn.Softmax())        

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        # print(sequence_output.shape) # torch.Size([32, 60, 768])

        #average = torch.mean(sequence_output, dim=1)
        #outputs = self.classify(average, targets) + outputs[2:]
        
        outputs = self.layer1(sequence_output)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        logits = self.layer4(outputs)
        
        outputs = (logits, )

        if targets is not None:
            outputs = logits, targets

        return outputs  # logits, (targets)
"""

 # tampoco converge
class BERTMHC_CNN(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)

        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)    
        self.conv2 = nn.Conv2d(6, 16, 5) 
        
        self.fc1 = nn.Linear(16*12*189, 100) 
        self.fc2 = nn.Linear(100, 10)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]  # [batch_size, 60, 768]       
        
        sequence_output = sequence_output.view(       # [batch_size, 1, 60, 768]     
                sequence_output.shape[0], 1, 
                sequence_output.shape[1], 
                sequence_output.shape[2])
        
        x = F.relu(self.conv1(sequence_output))  # [1, 60, 768] -> [6, 60, 768]
        x = self.pool(x)           # [6, 28, 28] -> [6, 14, 14]
        x = F.relu(self.conv2(x))  # [6, 14, 14] -> [16, 10, 10]
        x = self.pool(x)           # [16, 10, 10] -> [16, 5, 5]
        
        x = x.view(-1, 16*12*189) 
        
        x = F.relu(self.fc1(x))
        logits = F.softmax(self.fc2(x))
        
        outputs = (logits, )        

        if targets is not None:
            outputs = logits, targets

        return outputs  # logits, (targets)

