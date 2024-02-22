from typing import Union
import torch
import torch, torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging

#@markdown Defino el modelo: **AutoModel**
from typing import Union
import torch, torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import logging

class MyModel(nn.Module):
    """
    Sources:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bert/modeling_bert.py#L1520
    """
    def __init__(self, path='Narrativa/legal-longformer-base-4096-spanish',
                 num_labels=2,
                 pretrained=True,
                 max_tokens=300,
                 mismatch=False,
                 lenReg=1,
                 pDropout=0.1,
                 uncased=True,
                 label_smoothing=0):
    # def __init__(self, path:str, num_labels, pretrained=True, max_tokens=None, mismatch=False, lenReg:float=0, pDropout=0.1, uncased=True, label_smoothing=0.0):
        super().__init__()
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lenReg=lenReg

        #config
        other_config = {'attention_probs_dropout_prob':pDropout, "hidden_dropout_prob":pDropout}
        config = AutoConfig.from_pretrained(path,**other_config)
        # num_labels = config.num_labels
        self.max_tokens = max_tokens if max_tokens else config.max_length
        #BASE
        logging.set_verbosity_error() #Ignore unused weights warning. (this model is for finetunning)
        if pretrained:  self.baseModel = AutoModel.from_pretrained(path, **other_config)
        else:           self.baseModel = AutoModel.from_config(config)
        logging.set_verbosity_warning()
        self._tokenizer = AutoTokenizer.from_pretrained(path, do_lower_case=uncased)
        #HEAD
        self.dropout = nn.Dropout(pDropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        #loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.to(self.device) #model.to(device)

    def tokenizer(self, txt:Union[str, list]):
        features = self._tokenizer(txt, padding=True, truncation=True, max_length=self.max_tokens, return_tensors='pt')
        return features
    def untokenize(self, features):
        input_ids = features['input_ids'].tolist()
        txt = []
        for id_sample in input_ids:
            tokens = self._tokenizer.convert_ids_to_tokens(id_sample[1:-1])#ignore bos, eos
            txt.append( self._tokenizer.convert_tokens_to_string(tokens) )
        return txt
    def forward(self, **input:dict):
        input={k:None if v is None else v.to(self.device) for k, v in input.items()}
        outputs = self.baseModel(**input)
        #ouptuts[0] = hidden_states
        #ouptuts[1] = pooler_output
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits        = self.classifier(pooled_output)
        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            )
    def predict(self, batch):
        outputs = self(**batch)
        prediction = outputs.logits
        return torch.nn.functional.softmax(prediction,-1)

    def loss(self, batch, labels:list):
        output = self(**batch)
        logits = output.logits.view(-1, self.num_labels)
        labels = torch.tensor([labels], device=self.device).view(-1) #ignore batching dim
        loss = self.criterion(logits, labels)

        if self.lenReg > 1E-6: #normalize by length
            factor=batch['attention_mask'].sum() #nElements
            loss = loss*self.lenReg*factor
        return loss


class DementiaPrediction:
    """The torch model to predict dementia diseases"""
    def __init__(self):
        """Loads the pytorch model"""
        # az_model=None
        # ftdbv_model=None

        az_model = MyModel()
        az_data = torch.load('dementia_prediction_app/.model/AD_CTR_longformer.pt', map_location=az_model.device)
        az_model.load_state_dict(az_data)
        ftd_model = MyModel()
        ftd_data = torch.load('dementia_prediction_app/.model/FTD_CTR_longformer.pt', map_location=ftd_model.device)
        ftd_model.load_state_dict(ftd_data)

        self.models = {'AZ':az_model, 'FTDbv':ftd_model}
        self.biases = {'AZ':0.84, 'FTDbv':0.92}


    def show_models(self):
        return self.models.keys()
    
    
    def _predict(self, sample:str, model):
        with torch.no_grad():
            input = model.tokenizer(sample)
            model.eval()
            predictions = model.predict(input)
            # bSample = model.untokenize(input)

            prediction = torch.mean(predictions,0)
            prob_1=prediction[1]
        return prob_1

    def predict(self,sample:str, allow_biased:bool):
        result = {}
        for key in self.models.keys():
            model = self.models[key]
            if model == None: continue
            prob = self._predict(sample, model)
            if allow_biased: result[key] = prob<self.biases[key]
            else:            result[key] = prob<0.5
        return result

    def random_sample(self, test_dataset, srcField, tgtField):
        sampling='index' #@param ['CTR','AD', 'index']
        byIndex = 8 #@param {type:'integer'}
        byIndex = int(byIndex)
        threshold=0.5 #@param {type:'number'}
        color_good=2 #@param {type:'integer'}
        color_bad=3   #@param {type:'integer'}
        if sampling == 'index':
            sample = test_dataset.iloc[byIndex][srcField]
            true_tgt = test_dataset.iloc[byIndex][tgtField]
        else:
            sample = test_dataset[test_dataset[tgtField]==true_tgt].sample(n=1)[srcField].iloc[0]

        return sample

