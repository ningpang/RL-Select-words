import os
import torch
import copy
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from model.encoder.bert_encoder import Bert_Encoder
from model.classifier.softmax_classifier import Softmax_Layer
from model.reinforcement.policy_net import Policy_Net
from torch.distributions.utils import probs_to_logits

class Selector(object):
    def __init__(self, config, traindata, ckpt=None):
        super(Selector, self).__init__()
        self.traindata = traindata
        self.config = config
        self.encoder = Bert_Encoder(config=config).to(config.device)
        self.classifier = Softmax_Layer(input_size=self.encoder.output_size, num_class=config.num_of_relation).to(config.device)
        if ckpt is not None:
            self.encoder.load_state_dict(torch.load(os.path.join(ckpt, 'encoder.pt')))
            self.classifier.load_state_dict(torch.load(os.path.join(ckpt, 'classifier.pt')))
        self.policy = Policy_Net(config).to(config.device)
        # the lr of different layers of policy
        self.optimizer = optim.Adam([{'params': self.policy.encoder.parameters(), 'lr': 0.00001},
                                     {'params': self.policy.fc.parameters(), 'lr': 0.001}])

    def batch_select_action(self, tokens, probs, lengthes, scope, rand_flag=False, eps_flag=False, eps_value=0.1):
        batch_action_probs = probs.detach().cpu().numpy()
        batch_masks = torch.zeros_like(probs)
        batch_size, max_seq_len, action_size = probs.size()

        erased_tokens = torch.zeros_like(tokens).to(self.config.device)

        for batch_id, seq_len in enumerate(lengthes):
            token_num = 0
            for seq_id in range(seq_len):
                action_prob = batch_action_probs[batch_id, seq_id]
                if (seq_id>scope[0][batch_id]-1 and seq_id<scope[1][batch_id]+1) or \
                   (seq_id>scope[2][batch_id]-1 and seq_id<scope[3][batch_id]+1) or \
                    seq_id==0 or tokens[batch_id, seq_id]==102 or tokens[batch_id, seq_id]==1012:
                    action_id = 0
                else:
                    if rand_flag:
                        if eps_flag and random.random() < eps_value:
                            action_id = np.random.choice(action_size)
                        else:
                            action_id = int(np.random.choice(np.arange(action_size), p=action_prob))
                    else:
                        action_id = int(np.argmax(action_prob))
                batch_masks[batch_id, seq_id, action_id] = 1
                if action_id == 0:
                    erased_tokens[batch_id, token_num] = tokens[batch_id, seq_id]
                    token_num += 1

        return batch_masks, erased_tokens

    def bacth_calcu_action_probs(self, inputs, train_flag, ckpt=None):
        e11, e12, e21, e22 = [], [], [], []
        if ckpt is not None:
            self.policy.load_state_dict(torch.load(os.path.join(ckpt, 'policy.pt')))
        for i in range(inputs.size()[0]):
            tokens = inputs[i].cpu().numpy()
            e11.append(np.argwhere(tokens == 30522)[0][0])
            e12.append(np.argwhere(tokens == 30523)[0][0])
            e21.append(np.argwhere(tokens == 30524)[0][0])
            e22.append(np.argwhere(tokens == 30525)[0][0])
        if train_flag:
            self.policy.train()
            probs = self.policy(inputs)
        else:
            self.policy.eval()
            with torch.no_grad():
                probs = self.policy(inputs)
        return probs, (e11, e12, e21, e22)

    def batch_get_rewards(self, seq_action_masks, tokens, erased_tokens, lengthes, labels):
        seq_erase_actions = seq_action_masks[:, :, 1]
        seq_earase_num = seq_erase_actions.sum(-1)

        sparse_rewards = seq_earase_num/lengthes

        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            erased_probs = F.softmax(self.classifier(self.encoder(erased_tokens)), dim=-1)
            probs = F.softmax(self.classifier(self.encoder(tokens)), dim=-1)
            prediction_rewards = (torch.log(erased_probs)-torch.log(probs)).transpose(1, 0)
            labels = labels.view(1, -1)
            prediction_rewards = torch.gather(prediction_rewards, 0, labels).view(-1)

        total_rewards = prediction_rewards + 1.5*sparse_rewards
        return total_rewards

    def batch_interact(self, tokens, lengthes, labels, train_flag, rand_flag, eps_flag, eps_value):
        seq_action_probs, scope = self.bacth_calcu_action_probs(tokens, train_flag)
        seq_action_logps = probs_to_logits(seq_action_probs)

        batch_size = lengthes.size(0)
        batch_mean_rewards = torch.zeros_like(lengthes).float()
        mask_list = []
        reward_list = []

        sample_cnt = 5
        for sample in range(sample_cnt):
            # seq_action_masks: batch x seq_len x action_size
            seq_action_masks, erased_tokens = self.batch_select_action(tokens, seq_action_probs, lengthes, scope,
                               rand_flag=rand_flag, eps_flag=eps_flag, eps_value=eps_value)
            batch_rewards =  self.batch_get_rewards(seq_action_masks, tokens, erased_tokens, lengthes, labels)
            batch_mean_rewards += batch_rewards

            seq_action_rewards = batch_rewards.unsqueeze(1).unsqueeze(-1).expand_as(seq_action_masks)
            seq_action_rewards = seq_action_rewards.contiguous()

            mask_list.append(seq_action_masks)
            reward_list.append(seq_action_rewards)

        if sample_cnt>1 or batch_size > 1:
            batch_mean_rewards /= sample_cnt
            base_mean_rewards = batch_mean_rewards.unsqueeze(1).unsqueeze(-1).expand_as(seq_action_probs)

            for sample_id in range(sample_cnt):
                reward_list[sample_id] -= base_mean_rewards

        return seq_action_probs, seq_action_logps, mask_list, reward_list, batch_mean_rewards

    def batch_aggregate_loss(self, seq_action_logps, mask_list, reward_list):
        assert len(mask_list) == len(reward_list)
        seq_len, batch_size, action_size = seq_action_logps.size()
        sample_cnt = len(mask_list)
        loss_mat = torch.zeros_like(seq_action_logps)  # Size([seq_len, batch_size, action_size])

        for seq_action_masks, seq_action_rewards in zip(mask_list, reward_list):
            loss_mat += seq_action_logps * seq_action_masks * seq_action_rewards
        final_loss = - torch.sum(loss_mat) / (batch_size * sample_cnt)
        return final_loss

    def reinforcement_learning(self, rand_flag=True, eps_flag=True, eps_value=None):
        losses = []
        for step, (labels, lengthes, tokens) in enumerate(self.traindata):
            tokens = torch.stack([x.to(self.config.device) for x in tokens], dim=0)
            lengthes = lengthes.to(self.config.device)
            labels = labels.to(self.config.device)
            seq_action_probs, seq_action_logps, mask_list, reward_list, bacth_mean_rewards = \
                self.batch_interact(tokens, lengthes, labels, train_flag=True, rand_flag=rand_flag, eps_flag=eps_flag, eps_value=eps_value)
            loss = self.batch_aggregate_loss(seq_action_logps, mask_list, reward_list)
            losses.append(loss.item())
            self.policy.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f'RL loss is {np.array(losses).mean()}')

    def save_policy_ckpt(self):
        torch.save(self.policy.state_dict(), os.path.join(self.config.ckpt, 'policy.pt'))

    def get_compress_rate(self, batch_masks, lengthes):
        seq_erase_actions = batch_masks[:, :, 1]
        seq_earase_num = seq_erase_actions.sum(-1)
        compress_rate = (1 - seq_earase_num / lengthes).mean()
        return compress_rate

    def compress_text(self, inputs, lengthes, ckpt=None):
        probs, scope = self.bacth_calcu_action_probs(inputs, False, ckpt)

        batch_action_probs = probs.detach().cpu().numpy()
        batch_masks = torch.zeros_like(probs)
        tokens = torch.zeros_like(inputs).to(self.config.device)

        for batch_id, seq_len in enumerate(lengthes):
            token_num = 0
            for seq_id in range(seq_len):
                action_prob = batch_action_probs[batch_id, seq_id]
                if (seq_id>scope[0][batch_id]-1 and seq_id<scope[1][batch_id]+1) or\
                   (seq_id>scope[2][batch_id]-1 and seq_id<scope[3][batch_id]+1) or \
                    seq_id==0 or tokens[batch_id, seq_id]==102 or tokens[batch_id, seq_id]==1012:
                    action_id = 0
                else:
                    action_id = int(np.argmax(action_prob))
                batch_masks[batch_id, seq_id, action_id] = 1

                if action_id == 0:
                    tokens[batch_id, token_num] = inputs[batch_id, seq_id]
                    token_num += 1
        compress_rate = self.get_compress_rate(batch_masks, lengthes)
        # print(tokens[1])
        # seq_erase_actions = batch_masks[:, :, 1]
        # erased_tokens = copy.deepcopy(inputs)
        # erased_tokens[:][seq_erase_actions == 1] = 0
        return tokens, compress_rate







