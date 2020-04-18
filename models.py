"""Model classes"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import layers

from transformers import DistilBertModel

class DistilBERT(nn.Module):
    """DistilBERT model to classify news

    Based on the paper:
    DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
    by Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf
    (https://arxiv.org/abs/1910.01108)
    """
    def __init__(self, config, num_labels, use_img=True, img_size=512):
        super(DistilBERT, self).__init__()
        self.img_size = img_size
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.classifier = layers.DistilBERTClassifier(config.dim, num_labels,
                                                      drop_prob=config.seq_classif_dropout,
                                                      use_img=use_img,
                                                      img_size=img_size)

    def forward(self, input_idxs, atten_masks, images):
        con_x = self.distilbert(input_ids=input_idxs,
                                attention_mask=atten_masks)[0][:, 0]
        img_x = self.resnet18(images).view(-1, self.img_size)
        logit = self.classifier(con_x, img_x)
        log = F.log_softmax(logit, dim=1)

        return log