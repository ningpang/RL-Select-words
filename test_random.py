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
    # RL_selector = Selector(config, trainloader, config.ckpt)
    # cur_acc = framework.evaluate_model(config, encoder, classifier, validloader)
    # print(f'test acc before training:{cur_acc}')

    best_acc = 0.0
    for i in range(config.total_round):
        framework.train_random_model(config, encoder, classifier, trainloader, config.step1_epochs)
        cur_acc = framework.evaluate_model(config, encoder, classifier, validloader)
        print(f'current test acc:{cur_acc}')
        if cur_acc>best_acc:
            best_acc = cur_acc
    print('Best acc: ', best_acc)
