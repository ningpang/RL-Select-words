import os
import torch
import framework
from argparse import ArgumentParser
from config import Config
import framework
from data_loader.data_loader import data_sampler, get_data_loader
from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.softmax_classifier import Softmax_Layer
from model.reinforcement.RL import Selector

if __name__ == '__main__':
    parser = ArgumentParser(description='Reinforcement Learning')
    parser.add_argument('--config', default='config.ini')
    args = parser.parse_args()
    config = Config(args.config)
    config.batch_size_per_step = int(config.batch_size / config.gradient_accumulation_steps)

    sampler = data_sampler(config)
    training, valid, test = sampler.get_data()
    traindata, validdata, testdata = [], [], []
    for relation in training:
        traindata += training[relation]
        validdata += valid[relation]
        testdata += test[relation]

    trainloader = get_data_loader(config, traindata)
    validloader = get_data_loader(config, validdata, batch_size=1)
    testloader = get_data_loader(config, testdata, batch_size=1)

    encoder = Bert_Encoder(config=config).to(config.device)
    classifier = Softmax_Layer(input_size=encoder.output_size, num_class=len(sampler.id2rel)).to(config.device)
    RL_selector = Selector(config, trainloader, config.ckpt)

    best_acc = 0.0
    for i in range(8):
        print(f'RC Epoch--{i + 1}:')
        for j in range(3):
            print(f'---RL Epoch{j + 1}---:')
            RL_selector.reinforcement_learning(rand_flag=True, eps_flag=True, eps_value=0.1)
            RL_selector.save_policy_ckpt()
        framework.train_select_model(config, encoder, classifier, RL_selector, trainloader, config.step1_epochs)
        cur_acc = framework.evaluate_model(config, encoder, classifier, validloader)
        if cur_acc > best_acc:
            best_acc = cur_acc
            # RL_selector.save_policy_ckpt()
        print(f'current test acc:{cur_acc}')