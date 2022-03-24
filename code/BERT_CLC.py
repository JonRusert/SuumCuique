import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_transformers import *
#from sklearn.model_selection import train_test_split
import csv
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

####################################
# Class to represent communities.  
####################################
class CommunityLM:
    

    # __init__()
    # parameters:
    # community - name of trained model. 
    # LMPath - path to model
    # threshold - value (0 - 1.0) to determine if a text belongs to a community
    def __init__(self, community = 'AfricanAmerican', LMPath = '.', threshold = 0.85):
        if(community == 'AfricanAmerican'):
            model_name = LMPath + '/' + 'AfricanAmericanWOBPTPredEqualSingle.pt'
        elif(community == 'NativeAmerican'):
            model_name = LMPath + '/' + 'NativeAmericanExpandedPredEqualSingle.pt'
        elif(community == 'Hispanic'):
            model_name = LMPath + '/' + 'HispanicPredEqualSingle.pt'
        elif(community == 'Hawaii'):
            model_name = LMPath + '/' + 'HawaiiPredEqualSingle.pt'
        elif(community == 'SouthAsian'):
            model_name = LMPath + '/' + 'SouthAsianPredEqualSingle.pt'
        elif(community == 'Popular'):
            model_name = LMPath + '/' + 'PopularPredEqualSingle.pt'
        elif(community == 'All'):
            model_names = [LMPath + '/' + 'AfricanAmericanWOBPTPredEqualSingle.pt', LMPath + '/' + 'NativeAmericanExpandedPredEqualSingle.pt', LMPath + '/' + 'HispanicPredEqualSingle.pt', LMPath + '/' + 'HawaiiPredEqualSingle.pt', LMPath + '/' + 'SouthAsianPredEqualSingle.pt']
        else: 
            model_name = LMPath + '/' community + '.pt'

        self.community = community
        
        if(self.community != 'All'):
            if torch.cuda.is_available():
                self.model = torch.load(model_name)
                self.model = self.model.cuda()
            else:
                self.model = torch.load(model_name, map_location = 'cpu')
        else:
            self.models = []
            for model_name in model_names:
                if torch.cuda.is_available():
                    model = torch.load(model_name)
                    model = model.cuda()
                else:
                    model = torch.load(model_name, map_location = 'cpu')
                self.models.append(model)

                    
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.threshold = threshold


    # takes in text and returns whether it aligns with the community/probability of aligning
    def predict(self, text):
        com = ['[CLS]'] +  self.tokenizer.tokenize(text)  + ['[SEP]']
        com = com[:512]
        com[-1] = '[SEP]'

        # pad to 512
        cur_ids = tokenizer.convert_tokens_to_ids(com) + [0] * (MAX_LEN - len(com))
        cur_ids = cur_ids[:512]

        #print(cur_ids)

        cur_segs = [0] * len(com) + [0] * (MAX_LEN - len(com))
        cur_segs = cur_segs[:512]

        
        tokens_tensor = torch.tensor([cur_ids])
        attribute_tensors = torch.tensor([cur_segs])
        
        if(self.community != 'All'):

            outputs = self.model(tokens_tensor, token_type_ids=None, attention_mask=attribute_tensors)

            softmax = torch.nn.Softmax(dim=1)
            outputs_sm = softmax(outputs)
            cur_output = outputs_sm[0]

            out_prob = cur_output[0].detach().numpy()


            if(out_prob > self.threshold):
                out_pred = 1
            else:
                out_pred = 0

        else: # if all communities, need to check if aligns with ANY community, prob is highest alignment or first alignment > 0.85
            high_prob = 0
            out_pred = 0
            for model in self.models:
                outputs = model(tokens_tensor, token_type_ids=None, attention_mask=attribute_tensors)

                softmax = torch.nn.Softmax(dim=1)
                outputs_sm = softmax(outputs)
                cur_output = outputs_sm[0]

                out_prob = cur_output[0].detach().numpy()

                if(out_prob > high_prob):
                    high_prob = out_prob
                
                if(out_prob > self.threshold):
                    out_pred = 1
                    break 

            out_prob = high_prob
                            
        
        return out_pred, out_prob
    


# base model for classification
class BertForNextSentencePrediction(nn.Module):
    
    def __init__(self, config, num_labels = 2, dropout = 0.3):
        super(BertForNextSentencePrediction, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)#, output_all_encoded_layers=False)
        #print(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits



# trains model and stores trained model in model_name
# train_file lines should be in format: label \t text
def trainModel(train_file, model_name):


    # load in data
    input_ids = []
    attention_masks = []
    labels = []
    traincsv = csv.reader(open(train_file), delimiter = '\t')
    for cur in traincsv:

        if(cur[0] == '1'):
            label = 0
        else:
            label = 1

            
        # if only 1 comment included, use as initial comment and leave reply empty
        if(len(cur) == 2):
            init_com = ['[CLS]'] +  tokenizer.tokenize(cur[1])  + ['[SEP]']
            init_com = init_com[:300]
            init_com[-1] = '[SEP]'

            reply = []
        else:
            # to allow for room for next comment, limit initial comment to 300 words
            init_com = ['[CLS]'] +  tokenizer.tokenize(cur[1])  + ['[SEP]']
            init_com = init_com[:300]
            init_com[-1] = '[SEP]'
        
            reply = tokenizer.tokenize(cur[2]) + ['[SEP]']

        context = init_com + reply 
        context = context[:512]
        context[-1] = '[SEP]'

        # pad to 512
        cur_ids = tokenizer.convert_tokens_to_ids(context) + [0] * (MAX_LEN - len(context))
        cur_ids = cur_ids[:512]

        #print(cur_ids)

        cur_segs = [0] * len(init_com) + [1] * len(reply) + [0] * (MAX_LEN - len(context))
        cur_segs = cur_segs[:512]

        #print(cur_segs)


        if(len(cur_ids) != len(cur_segs)):
            print(cur)

        input_ids.append(cur_ids)
        attention_masks.append(cur_segs)
        labels.append(label)


    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)
    train_labels = torch.tensor(labels)

    batch_size = 4

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size= batch_size)




    # train model
    num_labels = 2
    dropout = 0.3
    config = BertConfig()
    model = BertForNextSentencePrediction(config, num_labels, dropout)

    if torch.cuda.is_available():
        print ("Using cuda")
        model = model.cuda()


    lrlast = .001
    lrmain = .00001
    optim1 = optim.Adam(
        [
            {"params":model.bert.parameters(),"lr": lrmain},
            {"params":model.classifier.parameters(), "lr": lrlast},

        ])

    optimizer = optim1

    criterion = nn.CrossEntropyLoss()

    train_loss_set = []
    epochs = 2

    for _ in trange(epochs, desc="Epoch"):
        model.train()

        tr_loss = 0
        nb_tr_steps = 0

        for step, batch in enumerate(train_dataloader):
            #batch = tuple(t.to(device) for t in batch)

            #b_input_ids, b_input_mask = batch
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            #print(b_input_ids)
            #print(b_input_mask)
            optimizer.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            loss = criterion(outputs, b_labels)
            loss.backward()


            optimizer.step()

            tr_loss += loss.sum()

            nb_tr_steps += 1

        print('Train loss: {}'.format(tr_loss/nb_tr_steps))




    torch.save(model, model_name + '.pt')


# tests model on model_name model and prints accuracy on test_file
# test_file should be in format: id \t label \t text 
def testModel(test_file, model_name):
    if torch.cuda.is_available():
        print ("Using cuda")
        model = torch.load(model_name + '.pt')
        model = model.cuda()
    else:
        print("Using cpu")
        model = torch.load(model_name + '.pt', map_location = 'cpu')

    

    # load in data
    input_ids = []
    attention_masks = []
    labels = []
    ids = []
    #hex_ids = False

    testcsv = csv.reader(open(test_file), delimiter = '\t')
    for cur in testcsv:
        if(cur[0] == ''):
            continue

        cur_id = int(cur.pop(0))
        
        
        if(cur[0] == '1'):
            label = 0
        else:
            label = 1


        # if only 1 comment included, use as initial comment and leave reply empty
        if(len(cur) == 2):
            init_com = ['[CLS]'] +  tokenizer.tokenize(cur[1])  + ['[SEP]']
            init_com = init_com[:300]
            init_com[-1] = '[SEP]'

            reply = []
        else:
            # to allow for room for next comment, limit initial comment to 300 words
            init_com = ['[CLS]'] +  tokenizer.tokenize(cur[1])  + ['[SEP]']
            init_com = init_com[:300]
            init_com[-1] = '[SEP]'
        
            reply = tokenizer.tokenize(cur[2]) + ['[SEP]']

        context = init_com + reply 
        context = context[:512]
        context[-1] = '[SEP]'

        # pad to 512
        cur_ids = tokenizer.convert_tokens_to_ids(context) + [0] * (MAX_LEN - len(context))
        cur_ids = cur_ids[:512]

        #print(cur_ids)

        cur_segs = [0] * len(init_com) + [1] * len(reply) + [0] * (MAX_LEN - len(context))
        cur_segs = cur_segs[:512]

        #print(cur_segs)


        if(len(cur_ids) != len(cur_segs)):
            print(cur)

        input_ids.append(cur_ids)
        attention_masks.append(cur_segs)
        labels.append(label)
        ids.append(cur_id)

    test_inputs = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_masks)
    test_labels = torch.tensor(labels)
    test_ids = torch.tensor(ids)

    batch_size = 8

    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_ids)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size= batch_size)

    

    model.eval()

    total = 0
    correct = 0
    pred_csv = csv.writer(open(test_file.split('.')[0] + '_' + model_name + '_preds.tsv', 'w'), delimiter = '\t')

    for step, batch in enumerate(test_dataloader):
        #batch = tuple(t.to(device) for t in batch)
        
        #b_input_ids, b_input_mask = batch
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_ids = batch[3].to(device)


        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        softmax = torch.nn.Softmax(dim=1)
        outputs_sm = softmax(outputs)
        
        #print(outputs_sm[0])
        for i in range(len(outputs_sm)):
            cur_output = outputs_sm[i]
            total += 1

            print(cur_output)

            if(cur_output[0] > cur_output[1]):
                pred = 0
                out_pred = 1
            else:
                pred = 1
                out_pred = 0
                
            if(pred == b_labels[i]):
                correct += 1

            out_pred = [b_ids[i].detach().numpy(), out_pred, cur_output[0].detach().numpy()]

            pred_csv.writerow(out_pred)
    
        
    print("Total correct:", correct, '/', total, '=', correct/total * 100, '%')


if(__name__ == "__main__"):
    if(sys.argv[3] == 'train'):
        trainModel(sys.argv[1], sys.argv[2])
    else:
        testModel(sys.argv[1], sys.argv[2])
