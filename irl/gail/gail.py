import chainer
import numpy as np
import chainer.functions as F
from chainerrl.agents import PPO, TRPO
from chainerrl.policies import SoftmaxPolicy
from itertools import chain


class GAIL(PPO):
    def __init__(self, discriminator, demonstrations, **kwargs):
        # super take arguments for dynamic inheritance
        super(self.__class__, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.demo_states = self.xp.asarray(np.asarray(list(chain(*demonstrations['states']))).astype(np.float32))
        self.demo_actions = self.xp.asarray(np.asarray(list(chain(*demonstrations['actions']))).astype(np.float32))

    def _update(self, dataset):
        # override func
        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)
        xp = self.xp

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.minibatch_size, shuffle=True)
        loss_mean = 0
        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = self.batch_states([b['state'] for b in batch], xp, self.phi)
            actions = xp.array([b['action'] for b in batch])

            demonstrations_indexes = np.random.permutation(len(self.demo_states))[:len(states)]
            demo_states, demo_actions = [d[demonstrations_indexes] for d in (self.demo_states, self.demo_actions)]

            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
                demo_states = self.obs_normalizer(demo_states, update=False)
            self.discriminator.train(self.convert_data_to_feed_discriminator(demo_states, demo_actions),
                                     self.convert_data_to_feed_discriminator(states, actions))
            loss_mean += self.discriminator.loss / (self.epochs * self.minibatch_size)
        super(self.__class__, self)._update(dataset)

    def _update_if_dataset_is_ready(self):
        # override func
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else
               sum(len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            # update reward in self.memory
            self._flush_last_episode()
            transitions = list(chain.from_iterable(self.memory))
            states = self.xp.asarray(np.concatenate([transition['state'][None] for transition in transitions]))
            actions = self.xp.asarray(np.concatenate([transition['action'][None] for transition in transitions]))
            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                rewards = self.discriminator.get_rewards(self.convert_data_to_feed_discriminator(states, actions)).array
            i = 0
            for episode in self.memory:
                for transition in episode:
                    transition['reward'] = float(rewards[i])
                    i += 1
            dataset = self._make_dataset()
            assert len(dataset) == dataset_size
            self._update(dataset)
            self.memory = []

    def convert_data_to_feed_discriminator(self, states, actions, noise_scale=0.1):
        xp = self.model.xp
        if isinstance(self.model.pi, SoftmaxPolicy):
            # if discrete action
            actions = xp.eye(self.model.pi.model.out_size, dtype=xp.float32)[actions.astype(xp.int32)]
        if noise_scale:
            actions += xp.random.normal(loc=0., scale=noise_scale, size=actions.shape)
        return F.concat((xp.array(states), xp.array(actions)))


def gailtype_constructor(rl_algo=TRPO):
    _gail_parent = GAIL.mro()[1]
    _gail_func_dict = {func: getattr(GAIL, func) for func in dir(GAIL) if callable(getattr(GAIL, func))
                       and (not func.startswith("__") or func == '__init__')
                       and (not hasattr(_gail_parent, func)
                            or not getattr(GAIL, func) == getattr(_gail_parent, func))}
    return type("GAIL" + rl_algo.__name__.upper(), (rl_algo,), _gail_func_dict)


# GAILTRPO do not work because TRPO's interface is not compatible with PPO
GAILTRPO = gailtype_constructor(rl_algo=TRPO)