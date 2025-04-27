import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, self.hidden)        
        self.fc4_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        x_critic = self.tanh(self.fc3_critic(x_critic))
        state_value = self.fc4_critic(x_critic)

        return normal_dist, state_value #It's returning action distribution and the value of the state estimated by the critic


class Agent(object):
    def __init__(self, policy, device='cpu', baseline = 0, actor_critic = False):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.actor_critic = actor_critic

        self.baseline = baseline
        self.gamma = 0.99
        
        self.states = []
        self.next_states = []

        self.state_values = []
        self.next_state_values = []

        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):

        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)

        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        done = torch.Tensor(self.done).to(self.train_device)


        returns = discount_rewards(rewards,self.gamma)

        #Empty lists to prepare for next episode

        #
        # TASK 2:
        # 1. Analyze the performance of the trained policies in terms of reward and time consumption.
            #With the same number of iterations model with baseline is being trained faster than the one without baseline.
        # 2. How would you choose a good value for the baseline?
            #The baseline should be chosen in a way that it is close to the expected return of the policy of last k episodes (use a moving average). 
            #If it's too low it's useless, if too high it will be too conservative and policy will not be able to learn enough.
        # 3. How does the baseline affect the training, and why?
            #It reduces the variance of the policy gradient estimates, making training faster.
            #It does not modify the expected value of the policy gradient (proof: grad(J) = E(grad(log(pi(a|s))) * (G_t - baseline)) = E(grad(log(pi(a|s)))*G_t)-b*E(grad(log(pi(a|s)))) = ... - 0 since expected value of the gradient of a probability is 0.
            #On the opposite it reduces the variance because it is centering all episodic returns around a value (e.g. 0 if baseline is the average
        if self.actor_critic == False:
            #   - compute discounted returns    
            returns = returns - self.baseline
            #   - compute policy gradient loss function given actions and returns
            loss = - (action_log_probs*returns).mean()

        # TASK 3:
        if self.actor_critic:
            values = torch.stack(self.state_values).squeeze(-1)
            future_values = torch.stack(self.next_state_values).squeeze(-1)        
        #   - compute boostrapped discounted return estimates
            bootstraped_returns = returns + self.gamma * future_values * (1 - done)
        #   - compute advantage terms
            advantages = bootstraped_returns - values
        #   - compute actor loss and critic loss
            actor_loss = -(action_log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, bootstraped_returns)
            loss = actor_loss + critic_loss

    # compute gradients and step the optimizer     
          
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.states, self.next_states, self.action_log_probs, self.rewards, self.done, self.state_values, self.next_state_values = [], [], [], [], [], [], []

        return loss.item



    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, state_value = self.policy(x)
        value = state_value.squeeze(-1)

        if evaluation:  # Return mean
            if self.actor_critic:
                return normal_dist.mean, None, value
            else:
                return normal_dist.mean, None, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Computes Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()
            
            if self.actor_critic:
                return action, action_log_prob, value
            else:
                return action, action_log_prob, None


    def store_outcome(self, state, next_state, action_log_prob, reward, done, value = None, next_value = None):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
        if self.actor_critic:
            self.state_values.append(value.detach().view(1))
            self.next_state_values.append(next_value.detach().view(1))

