import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size

        self.obs_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([max_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([max_size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)

        self.cur_pointer = 0
        self.size = 0

    def add(self, replays: list):
        for rep in replays:
            obs, act, rew, next_obs, done = rep

            self.obs_buf[self.cur_pointer] = obs
            self.next_obs_buf[self.cur_pointer] = next_obs
            self.acts_buf[self.cur_pointer] = act
            self.rews_buf[self.cur_pointer] = rew
            self.done_buf[self.cur_pointer] = done

            self.cur_pointer = (self.cur_pointer + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        assert self.size >= batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def sample_batches(self, batch_size: int, n_batches: int):
        n_transitions = batch_size * n_batches
        idxs = np.random.randint(0, self.size, size=n_transitions)
        out_batches = [
            dict(
                obs=self.obs_buf[idxs[s:e]],
                next_obs=self.next_obs_buf[idxs[s:e]],
                acts=self.acts_buf[idxs[s:e]],
                rews=self.rews_buf[idxs[s:e]],
                done=self.done_buf[idxs[s:e]],
            )
            for s, e in zip(
                range(0, n_transitions, batch_size),
                range(batch_size, (n_batches + 1) * batch_size, batch_size),
            )
        ]
        assert len(out_batches) == n_batches
        return out_batches

    def sample_batch_obss(self, batch_size: int):
        assert self.size >= batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.obs_buf[idxs]

    def save_buffer(self, save_path):
        ckpt_dict = {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "max_size": self.max_size,
            "cur_pointer": self.cur_pointer,
            "size": self.size,
            "obs_buf": self.obs_buf,
            "next_obs_buf": self.next_obs_buf,
            "acts_buf": self.acts_buf,
            "rews_buf": self.rews_buf,
            "done_buf": self.done_buf,
        }
        torch.save(ckpt_dict, save_path)

    def load_buffer(self, load_path):
        ckpt_dict = torch.load(load_path)
        assert self.obs_dim == ckpt_dict["obs_dim"]
        assert self.act_dim == ckpt_dict["act_dim"]
        assert self.max_size == ckpt_dict["max_size"]

        self.cur_pointer = ckpt_dict["cur_pointer"]
        self.size = ckpt_dict["size"]

        self.obs_buf = ckpt_dict["obs_buf"]
        self.next_obs_buf = ckpt_dict["next_obs_buf"]
        self.acts_buf = ckpt_dict["acts_buf"]
        self.rews_buf = ckpt_dict["rews_buf"]
        self.done_buf = ckpt_dict["done_buf"]
