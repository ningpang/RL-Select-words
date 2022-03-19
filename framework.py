import torch
import random
import copy
import torch.nn as nn
import numpy as np
import torch.optim as optim
from transformers import BertTokenizer

def train_model(config, encoder, classifier, train_data, num_epochs):

    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': encoder.parameters(), 'lr': 0.00001},
                            {'params': classifier.parameters(), 'lr': 0.001}])

    for epoch in range(num_epochs):
        losses = []
        for step, (labels, _,tokens) in enumerate(train_data):
            encoder.zero_grad()
            classifier.zero_grad()

            labels = labels.to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            reps = encoder(tokens)
            logits = classifier(reps)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()
        print(f'Finetuning loss is {np.array(losses).mean()}')

def evaluate_model(config, encoder, classifier, test_data):
    encoder.eval()
    classifier.eval()
    n = len(test_data)
    correct = 0
    for step, (labels, _, tokens) in enumerate(test_data):
        labels = labels.to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)
        seen_sim = logits.cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)
        label_sim = logits[:, labels].cpu().data.numpy()
        if label_sim >= max_smi:
            correct += 1
    return correct/n

def train_select_model(config, encoder, classifier, selector, train_data, num_epochs):

    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': encoder.parameters(), 'lr': 0.00001},
                            {'params': classifier.parameters(), 'lr': 0.001}])

    tokenizer = BertTokenizer.from_pretrained(config.bert_path,
                                              additional_special_tokens=['[E11]', '[E12]', '[E21]', '[E22]'])

    for epoch in range(num_epochs):
        losses = []
        com_rates = []
        for step, (labels, lengthes, tokens) in enumerate(train_data):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            lengthes = lengthes.to(config.device)

            tokens, com_rate = selector.compress_text(tokens, lengthes, config.ckpt)

            encoder.zero_grad()
            classifier.zero_grad()
            reps = encoder(tokens)
            logits = classifier(reps)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            com_rates.append(com_rate.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()

        print(f'Erased/Original is {np.array(com_rates).mean()}')
        print(f'Finetuning loss is {np.array(losses).mean()}')

def random_erase_tokens(rate, inputs, lengthes):
    e11, e12, e21, e22 = [], [], [], []
    mask = torch.zeros_like(inputs)
    for i in range(inputs.size()[0]):
        tokens = inputs[i].cpu().numpy()
        e11.append(np.argwhere(tokens == 30522)[0][0])
        e12.append(np.argwhere(tokens == 30523)[0][0])
        e21.append(np.argwhere(tokens == 30524)[0][0])
        e22.append(np.argwhere(tokens == 30525)[0][0])
    for batch_id, seq_len in enumerate(lengthes):
        for seq_id in range(seq_len):
            if (seq_id > e11[batch_id] - 1 and seq_id < e12[batch_id] + 1) \
                    or (seq_id > e21[batch_id] - 1 and seq_id < e22[batch_id] + 1) \
                    or seq_id == 0 or seq_id == 1012 or seq_id== 102:
                mask[batch_id][seq_id] = 0
            else:
                r = random.random()
                if r > rate:
                    mask[batch_id][seq_id] = 1
                else:
                    mask[batch_id][seq_id] = 0
    erased_tokens = copy.deepcopy(inputs)
    erased_tokens[:][mask == 1] = 0
    return erased_tokens

def train_random_model(config, encoder, classifier, train_data, num_epochs):
    encoder.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': encoder.parameters(), 'lr': 0.00001},
                            {'params': classifier.parameters(), 'lr': 0.001}])

    tokenizer = BertTokenizer.from_pretrained(config.bert_path,
                                              additional_special_tokens=['[E11]', '[E12]', '[E21]', '[E22]'])

    for epoch in range(num_epochs):
        losses = []

        for step, (labels, lengthes, tokens) in enumerate(train_data):
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            labels = labels.to(config.device)
            lengthes = lengthes.to(config.device)
            tokens = random_erase_tokens(0.3525, tokens, lengthes)
            encoder.zero_grad()
            classifier.zero_grad()
            reps = encoder(tokens)
            logits = classifier(reps)
            loss = criterion(logits, labels)
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.max_grad_norm)
            optimizer.step()

        print(f'Finetuning loss is {np.array(losses).mean()}')
