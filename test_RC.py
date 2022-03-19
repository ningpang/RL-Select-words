import os
import torch
from argparse import ArgumentParser
from config import Config
import framework
from data_loader.data_loader import data_sampler, get_data_loader
from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.softmax_classifier import Softmax_Layer

if __name__ == '__main__':
    parser = ArgumentParser(description='Relation Classification')
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
    classifier = Softmax_Layer(input_size=encoder.output_size, num_class=config.num_of_relation).to(config.device)

    ckpt = config.ckpt
    encoder.load_state_dict(torch.load(os.path.join(ckpt, 'encoder.pt')))
    classifier.load_state_dict(torch.load(os.path.join(ckpt, 'classifier.pt')))

    cur_acc = framework.evaluate_model(config, encoder, classifier, testloader)
    print(cur_acc)