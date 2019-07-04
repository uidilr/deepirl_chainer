import chainer
import chainer.functions as F
from irl.common.model import MLP
from chainer.link_hooks.spectral_normalization import SpectralNormalization


class Discriminator:
    def __init__(self, n_layer=4, n_units=32):
        self.reward_net = MLP(n_layer, n_units, 1, hook=SpectralNormalization, hook_params=dict(factor=1))
        self.value_net = MLP(n_layer, n_units, 1)  # , hook=SpectralNormalization, hook_params=dict(factor=10))
        # adding spectral normalization with small factor for value net makes training unstable
        # but why adversarial loss decreases when we add this to the value net?
        # because the lipschitz factor can be bounded by the lipschitz constant of the reward net?
        self.reward_optimizer = chainer.optimizers.Adam()
        self.value_optimizer = chainer.optimizers.Adam()
        self.reward_optimizer.setup(self.reward_net)
        self.value_optimizer.setup(self.value_net)

    def __call__(self, x):
        return self.reward_net(x), self.value_net(x)

    def train(self, expert_states, expert_next_states, expert_action_probs, fake_states, fake_next_states,
              fake_action_probs, gamma):

        def logits(states, next_states, log_action_probs):
            # p(expert|state, action) = sigmoid(logits)
            rewards = self.reward_net(states)
            # print(F.mean(rewards))
            state_values = self.value_net(states)
            next_state_values = self.value_net(next_states)
            return rewards + gamma * next_state_values - state_values - log_action_probs[:, None].array

        # This parameter stabilise training
        # softplus(logits) == log(sigmoid(logits))
        # print('expert: ', end='')
        loss = F.mean(F.softplus(-logits(expert_states, expert_next_states, expert_action_probs)))
        # print('fake: ', end='')
        loss += F.mean(F.softplus(logits(fake_states, fake_next_states, fake_action_probs)))

        # add gradient penalty for reward
        # xp = chainer.cuda.get_array_module(expert_states)
        # e = xp.random.uniform(0., 1., len(expert_states))[:, None].astype(xp.float32)
        # x_hat = chainer.Variable((e * expert_states + (1 - e) * fake_states), requires_grad=True)
        # grad, = chainer.grad([self.reward_net(x_hat)], [x_hat], enable_double_backprop=True)
        # loss_grad = 0.1 * F.mean(F.sqrt(F.batch_l2_norm_squared(grad)))
        # loss += loss_grad

        self.reward_net.cleargrads()
        self.value_net.cleargrads()
        loss.backward()
        self.reward_optimizer.update()
        self.value_optimizer.update()
        return loss

    def get_rewards(self, x):
        return self.reward_net(x)
