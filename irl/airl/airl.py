import chainer
import numpy as np
from chainerrl.agents import PPO
from itertools import chain
import copy
import collections

from irl.airl.discriminator import Discriminator
from irl.common.utils.mean_or_nan import mean_or_nan


class AIRL(PPO):
    def __init__(self, discriminator: Discriminator, demonstrations, discriminator_loss_stats_window=1000, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        demo_states = demonstrations['states']
        demo_actions = demonstrations['actions']
        demo_next_states = copy.deepcopy(demo_states)

        for demo_states_epi, demo_action_epi, demo_next_state_epi in zip(demo_states, demo_actions, demo_next_states):
            # delete last state because there is no next state of last state
            del demo_states_epi[-1]
            # delete action at last state because there is no next state after the last action is taken
            del demo_action_epi[-1]
            # delete first state so that deepcopy of demo_states will be the next states of the demo_state
            del demo_next_state_epi[0]
        self.demo_states = np.array(list(chain(*demo_states)))
        self.demo_next_states = np.array(list(chain(*demo_next_states)))
        self.demo_actions = np.array(list(chain(*demo_actions)))

        self.discriminator_loss_record = collections.deque(maxlen=discriminator_loss_stats_window)
        self.reward_mean_record = collections.deque(maxlen=discriminator_loss_stats_window)

    def _update(self, dataset):
        # override func
        xp = self.xp

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        dataset_iter = chainer.iterators.SerialIterator(dataset, self.minibatch_size, shuffle=True)
        loss_mean = 0
        while dataset_iter.epoch < self.epochs:
            # create batch for this iter
            batch = dataset_iter.__next__()
            states = self.batch_states([b['state'] for b in batch], xp, self.phi)
            next_states = self.batch_states([b['next_state'] for b in batch], xp, self.phi)
            actions = xp.array([b['action'] for b in batch])

            # create batch of expert data for this iter
            demonstrations_indexes = np.random.permutation(len(self.demo_states))[:self.minibatch_size]
            demo_states, demo_actions, demo_next_states = [d[demonstrations_indexes]
                                                           for d in (self.demo_states, self.demo_actions,
                                                                     self.demo_next_states)]

            states, demo_states, next_states, demo_next_states = [(self.obs_normalizer(d, update=False)
                                                                  if self.obs_normalizer else d)
                                                                  for d in [states, demo_states,
                                                                            next_states, demo_next_states]]

            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                action_log_probs = self.get_probs(states, actions)
                demo_action_log_probs = self.get_probs(demo_states, demo_actions)

            loss = self.discriminator.train(expert_states=demo_states, expert_next_states=demo_next_states,
                                            expert_action_probs=demo_action_log_probs, fake_states=states,
                                            fake_next_states=next_states, fake_action_probs=action_log_probs,
                                            gamma=self.gamma)
            loss_mean += loss / (self.epochs * self.minibatch_size)
        self.discriminator_loss_record.append(float(loss_mean.array))
        super()._update(dataset)

    def _update_if_dataset_is_ready(self):
        # override func
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else sum(
                len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            self._flush_last_episode()

            # update reward in self.memory
            transitions = list(chain(*self.memory))
            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                rewards = self.discriminator.get_rewards(np.concatenate([transition['state'][None]
                                                                         for transition in transitions])).array
            self.reward_mean_record.append(float(np.mean(rewards)))
            i = 0
            for episode in self.memory:
                for transition in episode:
                    transition['reward'] = float(rewards[i])
                    i += 1
            assert self.memory[0][0]['reward'] == float(rewards[0]), 'rewards is not replaced.'

            dataset = self._make_dataset()
            assert len(dataset) == dataset_size
            self._update(dataset)
            self.memory = []

    def get_probs(self, states, actions):
        target_distribs, _ = self.model(states)
        return target_distribs.log_prob(actions)

    def get_statistics(self):
        return [('average_discriminator_loss', mean_or_nan(self.discriminator_loss_record)),
                ('average_rewards', mean_or_nan(self.reward_mean_record))] + super().get_statistics()




