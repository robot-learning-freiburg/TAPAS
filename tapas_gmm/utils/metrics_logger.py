from collections import deque

import wandb


class MetricsLogger:
    def __init__(self):
        self.total_successes = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.total_cor_steps = 0
        self.total_pos_steps = 0
        self.total_neg_steps = 0
        self.episode_metrics = deque(maxlen=1)
        self.reset_episode()
        return

    def reset_episode(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_cor_steps = 0
        self.episode_pos_steps = 0
        self.episode_neg_steps = 0
        return

    def log_step(self, reward, feedback):
        self.episode_reward += reward
        self.episode_steps += 1
        if feedback == -1:
            self.episode_cor_steps += 1
        elif feedback == 1:
            self.episode_pos_steps += 1
        elif feedback == 0:
            self.episode_neg_steps += 1
        else:
            raise NotImplementedError
        return

    def log_episode(self, current_episode):
        episode_metrics = {
            "reward": self.episode_reward,
            "ep_cor_rate": self.episode_cor_steps / self.episode_steps,
            "ep_pos_rate": self.episode_pos_steps / self.episode_steps,
            "ep_neg_rate": self.episode_neg_steps / self.episode_steps,
            "episode": current_episode,
        }
        self.append(episode_metrics)
        self.total_episodes += 1
        if self.episode_reward > 0:
            self.total_successes += 1
        self.total_steps += self.episode_steps
        self.total_cor_steps += self.episode_cor_steps
        self.total_pos_steps += self.episode_pos_steps
        self.total_neg_steps += self.episode_neg_steps
        self.reset_episode()
        return

    def log_session(self):
        success_rate = self.total_successes / self.total_episodes
        cor_rate = self.total_cor_steps / self.total_steps
        pos_rate = self.total_pos_steps / self.total_steps
        neg_rate = self.total_neg_steps / self.total_steps
        wandb.run.summary["success_rate"] = success_rate
        wandb.run.summary["total_cor_rate"] = cor_rate
        wandb.run.summary["total_pos_rate"] = pos_rate
        wandb.run.summary["total_neg_rate"] = neg_rate
        return

    def append(self, episode_metrics):
        self.episode_metrics.append(episode_metrics)
        return

    def pop(self):
        return self.episode_metrics.popleft()

    def empty(self):
        return len(self.episode_metrics) == 0
