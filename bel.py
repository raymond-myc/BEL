# built-in libs
import argparse
import copy
import datetime
from distutils import util
import os
import time
from timeit import default_timer as timer
from typing import Union
import random

# third-party libs
import gym
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# custom libs
from buffers import ReplayBuffer
from helpers import (
    calc_acts_delta,
    partitioned_sample_uni,
    rank_tensor_elements,
    setup_std_logger,
    setup_tb_logger,
)
from networks import (
    ActorPack,
    CriticPack,
    calc_model_param_sensitivity,
    MlpActor,
    MlpCritic,
    net_to_vec,
    net_from_vec,
    sample_behav_net,
    target_network_update,
)


class BEL:
    def __init__(
        self,
        exp_log_root_dir: str,
        seed: int,
        env_name: str,
        max_timesteps: int,
        warmup_expl_steps: int,
        warmup_train_steps: int,
        max_steps_per_episode: int,
        timesteps_per_epoch: int,
        batch_size: int,
        reg_std: float,
        alpha: float,
        n_offsprings: int,
        buffer_size: int,
        btt_max_delta: float,
        brp_max_delta: float,
        act_noise_scale: float,
        target_policy_noise_factor: float,
        target_policy_noise_clip: float,
        polyak_gamma: float,
        reward_discount: float,
        update_actor_freq: float,
        actor_lr: float,
        actor_optimizer: str,
        actor_mid_act: str,
        actor_hidden_dims: list,
        actor_layer_norm: bool,
        critic_lr: float,
        critic_optimizer: str,
        critic_mid_act: str,
        critic_hidden_dims: list,
    ):
        self.exp_log_root_dir = exp_log_root_dir
        self.seed = seed
        self.reg_std = reg_std
        self.alpha = alpha
        self.n_offsprings = n_offsprings

        self.buffer_size = buffer_size
        self.btt_max_delta = btt_max_delta
        self.brp_max_delta = brp_max_delta

        self.act_noise_scale = act_noise_scale
        self.target_policy_noise_factor = target_policy_noise_factor
        self.target_policy_noise_clip = target_policy_noise_clip
        self.polyak_gamma = polyak_gamma
        self.reward_discount = reward_discount
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.actor_layer_norm = actor_layer_norm
        self.update_actor_freq = update_actor_freq
        self.actor_mid_act = actor_mid_act
        self.actor_optimizer = actor_optimizer
        self.critic_mid_act = critic_mid_act
        self.critic_optimizer = critic_optimizer
        self.env_name = env_name
        self.max_timesteps = max_timesteps
        self.warmup_expl_steps = warmup_expl_steps
        self.warmup_train_steps = warmup_train_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.total_timesteps_per_epoch = timesteps_per_epoch
        self.batch_size = batch_size

        self.setup()

    def setup(self):
        # * Setup folders and loggers
        # setup experiment log directory
        self.exp_log_dir = os.path.join(
            self.exp_log_root_dir, f"{self.env_name}_seed{self.seed}"
        )
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.exp_log_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.std_logger = setup_std_logger(self.exp_log_dir, print_to_console=False)
        self.tb_logger = setup_tb_logger(self.exp_log_dir)
        self.std_logger.info(
            "> Logging info to {}".format(os.path.abspath(self.exp_log_dir))
        )
        self.std_logger.info(
            "{} Parameters {} \n {} \n {} Parameters {} \n".format(
                ">" * 8, "<" * 8, self.__dict__, ">" * 8, "<" * 8
            )
        )

        # * Setup env
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.env_act_low = self.env.action_space.low
        self.env_act_high = self.env.action_space.high
        self.tensor_env_act_low = torch.tensor(self.env_act_low.copy())
        self.tensor_env_act_high = torch.tensor(self.env_act_high.copy())

        # * Setup seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.env.seed(self.seed)

        # * Setup models
        if self.actor_mid_act == "relu":
            self.actor_mid_act = nn.ReLU
        elif self.actor_mid_act == "tanh":
            self.actor_mid_act = nn.Tanh
        else:
            raise Exception
        if self.actor_optimizer == "adam":
            self.actor_optimizer = torch.optim.Adam
        else:
            raise Exception
        if self.critic_mid_act == "relu":
            self.critic_mid_act = nn.ReLU
        else:
            raise Exception

        if self.critic_optimizer == "adam":
            self.critic_optimizer = torch.optim.Adam
        else:
            raise Exception

        remote_ReplayBuffer = ray.remote(ReplayBuffer)
        self.replay_buffer = remote_ReplayBuffer.remote(
            obs_dim=self.state_dim, act_dim=self.action_dim, max_size=self.buffer_size
        )
        self.actor_net_params = {
            "input_dim": self.state_dim,
            "output_dim": self.action_dim,
            "hidden_dims": self.actor_hidden_dims,
            "action_low": self.tensor_env_act_low,
            "action_high": self.tensor_env_act_high,
            "mid_act": self.actor_mid_act,
            "layer_norm": self.actor_layer_norm,
        }
        self.critic_net_params = {
            "input_dim": self.state_dim + self.action_dim,
            "output_dim": 1,
            "hidden_dims": self.critic_hidden_dims,
        }
        self.actor_template = MlpActor(**self.actor_net_params)
        self.actor_vec_len = len(net_to_vec(self.actor_template))

        self.center_net = MlpActor(**self.actor_net_params)
        self.center_vec = net_to_vec(self.center_net)
        self.brp = BRP(self.actor_template, self.actor_vec_len)
        ws = torch.Tensor(
            [
                np.log((self.n_offsprings + 1) / i)
                for i in range(1, self.n_offsprings + 1)
            ]
        )
        ws = ws / ws.sum()
        self.recomb_weights = ws

        self.rl_instances = []
        for offspring_id in range(self.n_offsprings):
            rl_actor_pack = self.new_actor_pack()
            rl_critic_pack = self.new_critic_pack()
            rl_ins = TD3.remote(
                offspring_id=offspring_id,
                reg_std=self.reg_std,
                alpha=self.alpha,
                actor_pack=rl_actor_pack,
                critic_pack=rl_critic_pack,
                replay_buffer=self.replay_buffer,
                batch_size=self.batch_size,
                update_actor_freq=self.update_actor_freq,
                polyak_gamma=self.polyak_gamma,
                reward_discount=self.reward_discount,
                tensor_env_act_low=self.tensor_env_act_low,
                tensor_env_act_high=self.tensor_env_act_high,
                target_policy_noise_clip=self.target_policy_noise_clip,
                target_policy_noise_factor=self.target_policy_noise_factor,
            )
            self.rl_instances.append(rl_ins)

        # * Setup global records
        self.total_timesteps = 0
        self.gradient_steps = 0
        self.actor_grad_steps = 0
        self.critic_grad_steps = 0
        self.epoch = 0
        self.generations = 0
        self.start_time = timer()
        self.eval_results = []
        self.generation_threshold = int(self.n_offsprings * self.max_steps_per_episode)
        self.grad_steps_before_generation = 0
        self.latest_test_max_rew = 0
        self.latest_test_mean_rew = 0

    def run(self):
        with tqdm(
            total=self.max_timesteps,
            desc="Env: {} Seed: {}".format(self.env_name, self.seed),
        ) as pbar:
            while self.flag_run():
                if self.flag_warmup():
                    if self.total_timesteps == 0:
                        self.std_logger.info("Warming up with random actions ...")
                    roll_result = self.rollout(
                        env=self.env, actor=None, action_noise=False
                    )
                    rolled_steps = roll_result["rolled_steps"]
                    self.total_timesteps += rolled_steps
                    episode_len = rolled_steps
                    ray.get(self.replay_buffer.add.remote(roll_result["replays"]))
                    pbar.update(rolled_steps)

                if self.flag_train():
                    # RL training
                    n_grad_steps = int(episode_len * self.n_offsprings)
                    train_tasks = [
                        self.rl_instances[offspring_id].train.remote(n_grad_steps)
                        for offspring_id in range(self.n_offsprings)
                    ]
                    start_train_time = time.time()
                    train_results = ray.get(train_tasks)
                    end_train_time = time.time()
                    self.std_logger.info(
                        "Training took {:.2f} seconds".format(
                            end_train_time - start_train_time
                        )
                    )
                    # Collect trained results
                    for res in train_results:
                        offspring_id = res["offspring_id"]
                        self.actor_grad_steps += res["actor_grad_steps"]
                        self.critic_grad_steps += res["critic_grad_steps"]
                        self.tb_logger.add_scalar(
                            f"Train/Loss/Actor/{offspring_id}",
                            res["actor_loss"],
                            self.total_timesteps,
                        )
                        self.tb_logger.add_scalar(
                            f"Train/Loss/Critic/{offspring_id}",
                            res["critic_loss"],
                            self.total_timesteps,
                        )
                        self.tb_logger.add_scalar(
                            f"Train/Q_value/{offspring_id}",
                            res["Q_value"],
                            self.total_timesteps,
                        )
                    self.grad_steps_before_generation += n_grad_steps

                    # * Interact with environment after training
                    local_rolled_steps = []
                    offspring_vecs = {}
                    offspring_rews = {}
                    for offspring_id in range(self.n_offsprings):
                        offspring_vec = ray.get(
                            self.rl_instances[offspring_id].get_actor_vec.remote()
                        )
                        offspring_net = net_from_vec(self.actor_template, offspring_vec)
                        offspring_vecs[offspring_id] = offspring_vec
                        res = self.rollout(
                            env=self.env,
                            actor=offspring_net,
                            action_noise=(self.act_noise_scale > 0),
                        )
                        rolled_steps = res["rolled_steps"]
                        self.total_timesteps += rolled_steps
                        local_rolled_steps.append(rolled_steps)
                        ep_replays = res["replays"]
                        ep_rew = res["ep_reward"]
                        offspring_rews[offspring_id] = ep_rew
                        ray.get(self.replay_buffer.add.remote(ep_replays))
                    self.std_logger.info(
                        "Episode rews of offsprings: {}".format(offspring_rews)
                    )
                    episode_len = int(np.mean(local_rolled_steps))
                    self.std_logger.info(
                        "Mean episode length this iteration: {}".format(episode_len)
                    )

                    # * Step to next generation when gradient steps meet threshold
                    if self.flag_new_gen():
                        self.generations += 1
                        self.std_logger.info(
                            "{} grad steps accumulated, met threshold of {}, step to generation {}".format(
                                self.grad_steps_before_generation,
                                self.generation_threshold,
                                self.generations,
                            )
                        )

                        # * Update population
                        self.tell(offspring_vecs, offspring_rews)

                        # * Ask new generation of offsprings and btt deltas
                        offsprings, btt_deltas = self.ask()
                        for offspring_id in range(self.n_offsprings):
                            offspring = offsprings[offspring_id]
                            asked_actor_pack = ActorPack(
                                actor=offspring,
                                target_actor=copy.deepcopy(offspring),
                                optimizer=self.actor_optimizer(
                                    offspring.parameters(), self.actor_lr
                                ),
                            )
                            self.rl_instances[offspring_id].set_mean_net.remote(
                                copy.deepcopy(self.center_net)
                            )
                            self.rl_instances[offspring_id].set_actor_pack.remote(
                                asked_actor_pack, btt_deltas[offspring_id]
                            )

                        # * Reset grad step record for this generation
                        self.grad_steps_before_generation = 0
                    else:
                        self.std_logger.info(
                            "{} grad steps accumulated, not met threshold of {}, still in generation {}".format(
                                self.grad_steps_before_generation,
                                self.generation_threshold,
                                self.generations,
                            )
                        )

                    # Update progress bar
                    pbar.update(np.sum(local_rolled_steps))

                if self.flag_test():
                    self.test()
                    pbar.set_postfix(
                        {
                            "Test max reward": "{:.2f}".format(
                                self.latest_test_max_rew
                            ),
                            "Test mean reward": "{:.2f}".format(
                                self.latest_test_mean_rew
                            ),
                        }
                    )

    def ask(self):
        """
        Initialize a generation of BRP-processed offsprings and a list of
        target BTT deltas.
        """
        batch_obss = ray.get(
            self.replay_buffer.sample_batch_obss.remote(self.batch_size)
        )
        batch_obss = torch.tensor(batch_obss.copy())
        mean_net_sens = calc_model_param_sensitivity(batch_obss, self.center_net)[
            "sens"
        ]
        brp_deltas = partitioned_sample_uni(
            n_samples=self.n_offsprings, upper_bound=self.brp_max_delta
        )
        offsprings = []
        s_time = time.time()
        for offspring_id in range(self.n_offsprings):
            offspring = self.brp.perturb_actor(
                batch_obss=batch_obss,
                actor_vec=self.center_vec,
                perturb_scale=brp_deltas[offspring_id],
                center_sens=mean_net_sens,
            )
            offsprings.append(offspring)
        e_time = time.time()
        btt_deltas = partitioned_sample_uni(
            n_samples=self.n_offsprings, upper_bound=self.btt_max_delta
        )
        return offsprings, btt_deltas

    def tell(self, offspring_vecs: dict, offspring_rews: dict):
        """
        Recombination of offsprings into a new center.
        """
        tensor_vecs = torch.stack([v for v in offspring_vecs.values()])
        tensor_rews = torch.tensor([v for v in offspring_rews.values()])
        rew_ranks = rank_tensor_elements(tensor_rews, descending=True)
        sorted_idxs = torch.argsort(rew_ranks, descending=False)
        sorted_vecs = tensor_vecs[sorted_idxs]
        self.center_vec = self.recomb_weights @ sorted_vecs
        self.center_net = net_from_vec(self.actor_template, self.center_vec)

    def test(self):
        self.std_logger.info(f"Testing policy performance with population mean ...")
        test_actor = copy.deepcopy(self.center_net)
        ep_rews = []
        for _ in range(10):
            res = self.rollout(
                env=self.env,
                actor=test_actor,
                action_noise=False,
            )
            ep_rews.append(res["ep_reward"])
        ep_rews = np.array(ep_rews)
        mean_rew = np.mean(ep_rews)
        max_rew = np.max(ep_rews)
        min_rew = np.min(ep_rews)
        self.latest_test_max_rew = max_rew
        self.latest_test_mean_rew = mean_rew
        self.std_logger.info(
            "|Timesteps: {:>9.2E} | Generations: {:>5d} | CriticSteps: {:>9.2E} | ActorSteps: {:>9.2E} |\n|TestMean: {:>9.2E} | TestMax: {:>9.2E} | TestMin: {:>9.2E} |\n|ClockTime: {} |\n".format(
                self.total_timesteps,
                self.generations,
                self.critic_grad_steps,
                self.actor_grad_steps,
                mean_rew,
                max_rew,
                min_rew,
                datetime.timedelta(seconds=int(timer() - self.start_time)),
            )
        )
        self.tb_logger.add_scalar(f"Test/MeanReward", mean_rew, self.total_timesteps)
        self.tb_logger.add_scalar("Test/MaxReward", max_rew, self.total_timesteps)

        # * save results to csv
        self.eval_results.append([self.total_timesteps, mean_rew])
        eval_df = pd.DataFrame(self.eval_results, columns=["rollstep", "mean_rew"])
        eval_df.to_csv(os.path.join(self.exp_log_dir, "eval.csv"), index=None)

        # * save models to ckpt
        ckpt_path = os.path.join(self.models_dir, f"ckpt_{self.total_timesteps}.tar")
        ckpt_dict = {
            "timesteps": self.total_timesteps,
            "center_actor": self.center_net,
        }
        for i in range(self.n_offsprings):
            ckpt_dict[f"critic_{i}"] = ray.get(
                self.rl_instances[i].get_critic_net.remote()
            )
            ckpt_dict[f"actor_{i}"] = ray.get(
                self.rl_instances[i].get_actor_net.remote()
            )
        self.std_logger.info("Saving actor and critic models to {}".format(ckpt_path))
        torch.save(ckpt_dict, ckpt_path)
        return ep_rews

    def new_critic_pack(self):
        critic1 = MlpCritic(**self.critic_net_params)
        critic2 = MlpCritic(**self.critic_net_params)
        target_critic1 = copy.deepcopy(critic1)
        target_critic2 = copy.deepcopy(critic2)
        for T in [target_critic1, target_critic2]:
            for p in T.parameters():
                p.requires_grad = False
        critic_opt = self.critic_optimizer(
            list(critic1.parameters()) + list(critic2.parameters()),
            self.critic_lr,
        )
        return CriticPack(
            critic1=critic1,
            critic2=critic2,
            target_critic1=target_critic1,
            target_critic2=target_critic2,
            optimizer=critic_opt,
        )

    def new_actor_pack(self):
        actor = MlpActor(**self.actor_net_params)
        actor_pack = ActorPack(
            actor=actor,
            target_actor=copy.deepcopy(actor),
            optimizer=self.actor_optimizer(actor.parameters(), self.actor_lr),
        )
        return actor_pack

    def select_action(
        self,
        obs,
        actor,
        action_noise: bool,
    ):
        det_action = actor(torch.Tensor(obs.copy())).detach().cpu().numpy()
        if action_noise:
            aciton_noise = np.random.normal(
                loc=0, scale=self.act_noise_scale, size=det_action.shape
            )
            action = (det_action + aciton_noise).clip(
                min=self.env_act_low, max=self.env_act_high
            )
        else:
            action = det_action
        return action

    def rollout(
        self,
        env,
        actor: Union[nn.Module, None],
        action_noise: bool,
    ):
        replays = []
        done = False
        ep_rew = 0
        rolled_steps = 0
        cur_obs = env.reset()
        while not done:
            # * Select action
            if actor is None:
                action = env.action_space.sample()
            else:
                action = self.select_action(cur_obs, actor, action_noise)
            next_obs, reward, done, _ = env.step(action)
            rolled_steps += 1
            ep_rew += reward
            replay_done = done
            if rolled_steps == self.max_steps_per_episode:
                replay_done = False
            replay_tup = (cur_obs.copy(), action, reward, next_obs, replay_done)
            replays.append(replay_tup)
            cur_obs = next_obs
        return {
            "replays": replays,
            "rolled_steps": rolled_steps,
            "ep_reward": ep_rew,
            "actor": actor,
        }

    def flag_train(self):
        return self.total_timesteps >= self.warmup_train_steps

    def flag_run(self):
        return self.total_timesteps < self.max_timesteps

    def flag_warmup(self):
        return self.total_timesteps < self.warmup_expl_steps

    def flag_new_gen(self):
        return self.grad_steps_before_generation >= self.generation_threshold

    def flag_test(self):
        test_flag = False
        cur_epoch = self.total_timesteps // self.total_timesteps_per_epoch
        if cur_epoch > self.epoch:
            self.epoch += 1
            test_flag = True
        return self.flag_train() and test_flag


class BRP:
    def __init__(
        self,
        actor_template: nn.Module,
        actor_vec_len: int,
    ):
        self.actor_template = actor_template
        self.actor_vec_len = actor_vec_len

    def perturb_actor(
        self,
        batch_obss: torch.Tensor,
        actor_vec: torch.Tensor,
        perturb_scale: float,
        center_sens: torch.Tensor = None,
    ):
        center_net = net_from_vec(self.actor_template, actor_vec)
        if center_sens is None:
            center_sens = calc_model_param_sensitivity(batch_obss, center_net)["sens"]
        rand_dir = torch.randn(self.actor_vec_len)
        ba_amp = self.search_r(
            batch_obss,
            center_net,
            center_sens,
            target_delta=perturb_scale,
            init_r=0.1,
            dir_z=rand_dir,
        )
        ba_vec = ba_amp * rand_dir
        ba_net = sample_behav_net(center_net, 1, center_sens, ba_vec)
        return ba_net

    def search_r(
        self,
        batch_obss,
        target_actor,
        target_sens,
        target_delta,
        init_r,
        dir_z,
    ):
        ta_acts = target_actor(batch_obss)
        r = init_r
        delta = 0
        iters = 0
        while abs(target_delta - delta) > 5e-3:
            if delta < target_delta:
                r *= 1 / 0.8
            else:
                r *= 0.8
            ba = sample_behav_net(target_actor, r, target_sens, dir_z)
            ba_acts = ba(batch_obss)
            delta = calc_acts_delta(ba_acts, ta_acts)
            iters += 1
            if (r > 1e10) or (iters > 500):
                break
        return r


@ray.remote
class TD3:
    def __init__(
        self,
        reg_std: float,
        alpha: float,
        offspring_id: int,
        actor_pack: ActorPack,
        critic_pack: CriticPack,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        update_actor_freq: float,
        polyak_gamma: float,
        reward_discount: float,
        tensor_env_act_low: torch.Tensor,
        tensor_env_act_high: torch.Tensor,
        target_policy_noise_clip: float,
        target_policy_noise_factor: float,
    ) -> None:
        self.reg_std = reg_std
        self.alpha = alpha
        self.offspring_id = offspring_id
        self.actor_pack = actor_pack
        self.critic_pack = critic_pack
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_actor_freq = update_actor_freq
        self.polyak_gamma = polyak_gamma
        self.reward_discount = reward_discount
        self.target_policy_noise_clip = target_policy_noise_clip
        self.target_policy_noise_factor = target_policy_noise_factor
        self.tensor_env_act_high = tensor_env_act_high
        self.tensor_env_act_low = tensor_env_act_low

        # btt not activated for the first training iteration
        self.btt_delta = None

    def train(self, n_grad_steps: int):
        # * ------- setup -------
        if self.btt_delta is not None:
            behavior_regularizor = D.Normal(loc=self.btt_delta, scale=self.reg_std)
        A_losses = []
        Q_losses = []
        Q_value = []
        grad_step = 0
        critic_grad_steps = 0
        actor_grad_steps = 0
        critic1, critic2, target_critic1, target_critic2, critic_opt = (
            self.critic_pack.critic1,
            self.critic_pack.critic2,
            self.critic_pack.target_critic1,
            self.critic_pack.target_critic2,
            self.critic_pack.optimizer,
        )
        actor, target_actor, actor_opt = (
            self.actor_pack.actor,
            self.actor_pack.target_actor,
            self.actor_pack.optimizer,
        )
        if self.update_actor_freq >= 1:
            update_critic_freq = 1
            update_actor_freq = self.update_actor_freq
        else:
            update_critic_freq = int(1 / self.update_actor_freq)
            update_actor_freq = 1

        batches = []
        for grad_step in range(n_grad_steps):
            # As parallel training may make interacting with the buffer instance
            # inefficient, we implement a small local batch queue to maximize efficiency
            # this trick significantly boosts cpu utilization
            if not batches:
                batches = ray.get(
                    self.replay_buffer.sample_batches.remote(self.batch_size, 100)
                )
            batch = batches.pop()
            obss, acts, rews, next_obss, dones = [
                torch.tensor(item.copy())
                for item in (
                    batch["obs"],
                    batch["acts"],
                    batch["rews"],
                    batch["next_obs"],
                    batch["done"],
                )
            ]
            rews = rews.reshape(-1, 1)
            dones = dones.reshape(-1, 1)
            # * ------- critic step -------
            if grad_step % update_critic_freq == 0:
                sa_tup = torch.cat([obss, acts], dim=-1)
                q1 = critic1(sa_tup)
                Q_value.append(torch.mean(q1).item())
                q2 = critic2(sa_tup)
                with torch.no_grad():
                    ta = target_actor
                    pi_target = ta(next_obss)
                    noise_target = (
                        torch.randn_like(pi_target) * self.target_policy_noise_factor
                    ).clamp(
                        min=-self.target_policy_noise_clip,
                        max=self.target_policy_noise_clip,
                    )
                    pi_target = (pi_target + noise_target).clamp(
                        self.tensor_env_act_low,
                        self.tensor_env_act_high,
                    )
                    nsa_tup = torch.cat([next_obss, pi_target], dim=-1)
                    min_q_pi_target = torch.min(
                        target_critic1(nsa_tup), target_critic2(nsa_tup)
                    )
                    q_bellman_backup = (
                        rews + self.reward_discount * (1 - dones) * min_q_pi_target
                    )
                q1_loss = F.mse_loss(q1, q_bellman_backup)
                q2_loss = F.mse_loss(q2, q_bellman_backup)
                q_loss = q1_loss + q2_loss
                critic_opt.zero_grad()
                q_loss.backward()
                critic_opt.step()
                Q_losses.append(q_loss.detach().cpu().numpy())
                critic_grad_steps += 1
            # * ------- actor step -------
            if grad_step % update_actor_freq == 0:
                actor_acts = actor(obss)
                sa_pi_tup = torch.cat([obss, actor_acts], dim=-1)
                AC_loss = -critic1(sa_pi_tup).mean()
                if self.btt_delta is not None:
                    with torch.no_grad():
                        mean_net_acts = self.mean_net(obss)
                    BR_loss = -behavior_regularizor.log_prob(
                        calc_acts_delta(actor_acts, mean_net_acts)
                    )
                    A_loss = AC_loss + BR_loss * self.alpha
                else:
                    A_loss = AC_loss
                actor_opt.zero_grad()
                A_loss.backward()
                actor_opt.step()
                actor_grad_steps += 1
                A_losses.append(A_loss.detach().cpu().numpy())

                target_network_update(actor, target_actor, self.polyak_gamma)
                target_network_update(critic1, target_critic1, self.polyak_gamma)
                target_network_update(critic2, target_critic2, self.polyak_gamma)

        return {
            "offspring_id": self.offspring_id,
            "actor_loss": np.mean(A_losses),
            "critic_loss": np.mean(Q_losses),
            "Q_value": np.mean(Q_value),
            "actor_grad_steps": actor_grad_steps,
            "critic_grad_steps": critic_grad_steps,
        }

    def set_actor_pack(self, actor_pack: ActorPack, btt_delta: float):
        self.actor_pack = actor_pack
        self.btt_delta = btt_delta

    def set_critic_pack(self, critic_pack: CriticPack):
        self.critic_pack = critic_pack

    def set_mean_net(self, mean_net: nn.Module):
        self.mean_net = mean_net

    def get_actor_vec(self):
        actor_vec = net_to_vec(self.actor_pack.actor)
        return actor_vec

    def get_critic_net(self):
        return copy.deepcopy(self.critic_pack.critic1)

    def get_actor_net(self):
        return copy.deepcopy(self.actor_pack.actor)


if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()

    # Experiment specific params
    parser.add_argument("--exp_log_root_dir", default="results", type=str)
    parser.add_argument("--env_name", default="Ant-v3", type=str)
    parser.add_argument("--seed", default=0, type=int)

    # BEL specific params
    parser.add_argument("--n_offsprings", default=5, type=int)
    parser.add_argument("--btt_max_delta", default="1.0", type=float)
    parser.add_argument("--brp_max_delta", default=0.1, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--reg_std", default=0.1, type=float)

    # TD3 specific params
    parser.add_argument("--act_noise_scale", default="0.1", type=float)
    parser.add_argument("--buffer_size", default="1e6", type=lambda x: int(float(x)))
    parser.add_argument("--warmup_expl_steps", default=1e4, type=int)
    parser.add_argument("--warmup_train_steps", default=1e4, type=int)
    parser.add_argument("--max_steps_per_episode", default=1e3, type=int)
    parser.add_argument("--timesteps_per_epoch", default=1e4, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--rew_discount", default=0.99, type=float)
    parser.add_argument("--actor_optimizer", default="adam", type=str)
    parser.add_argument("--critic_optimizer", default="adam", type=str)
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_mid_act", default="relu", type=str)
    parser.add_argument(
        "--actor_hidden_dims",
        default=[256, 256],
        type=lambda s: [int(item) for item in s.split(",")],
    )
    parser.add_argument("--actor_layer_norm", default=False, type=util.strtobool)
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_mid_act", default="relu", type=str)
    parser.add_argument(
        "--critic_hidden_dims",
        default=[256, 256],
        type=lambda s: [int(item) for item in s.split(",")],
    )
    parser.add_argument("--update_actor_freq", default=1, type=float)
    parser.add_argument("--target_policy_noise_factor", default=0.2, type=float)
    parser.add_argument("--target_policy_noise_clip", default=0.5, type=float)
    parser.add_argument("--polyak_gamma", default=5e-3, type=float)
    args = parser.parse_args()

    bel = BEL(
        env_name=args.env_name,
        seed=args.seed,
        exp_log_root_dir=args.exp_log_root_dir,
        n_offsprings=args.n_offsprings,
        reg_std=args.reg_std,
        alpha=args.alpha,
        brp_max_delta=args.brp_max_delta,
        btt_max_delta=args.btt_max_delta,
        update_actor_freq=args.update_actor_freq,
        actor_lr=args.actor_lr,
        actor_mid_act=args.actor_mid_act,
        actor_optimizer=args.actor_optimizer,
        actor_hidden_dims=args.actor_hidden_dims,
        actor_layer_norm=args.actor_layer_norm,
        critic_lr=args.critic_lr,
        critic_mid_act=args.critic_mid_act,
        critic_optimizer=args.critic_optimizer,
        critic_hidden_dims=args.critic_hidden_dims,
        buffer_size=args.buffer_size,
        max_timesteps=args.max_timesteps,
        warmup_expl_steps=args.warmup_expl_steps,
        warmup_train_steps=args.warmup_train_steps,
        max_steps_per_episode=args.max_steps_per_episode,
        timesteps_per_epoch=args.timesteps_per_epoch,
        batch_size=args.batch_size,
        act_noise_scale=args.act_noise_scale,
        target_policy_noise_factor=args.target_policy_noise_factor,
        target_policy_noise_clip=args.target_policy_noise_clip,
        polyak_gamma=args.polyak_gamma,
        reward_discount=args.rew_discount,
    )
    bel.run()
