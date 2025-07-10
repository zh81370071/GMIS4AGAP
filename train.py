import hydra
import numpy as np
import os
import random
import shutil
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from copy import *
from itertools import chain
from tqdm import tqdm
from agent.solver import *
from agent.gnr.gnn_agent import *
from environment.knowledge import DispatchWrapper
import logging
import sys
from contextlib import redirect_stdout
import pickle


# ======================= 日志初始化 =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cliue_result.log", mode="w", encoding="utf-8"),  # 文件日志（覆盖写）
        logging.StreamHandler()                                        # 控制台同步输出
    ]
)
# =========================================================

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        # set seed
        self.set_seed(cfg.hyperparameters.seed)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def infer(self, solver, env, kwargs):
        # envs for validate
        envs = [env for _ in range(env.data.n_samples)]
        # batch of arguments
        args = [(solver, env, self.cfg, k, kwargs) for k, env in zip(range(env.data.n_samples), envs)]
        # wrappers
        # 循环执行 interact 函数
        wrappers = []
        experience = []
        with redirect_stdout(sys.stderr):  # 将输出重定向到stderr
            for arg in args:
                wrappers.append(interact(arg)[1])

        # sort the wrappers according to the data
        wrappers = sorted(wrappers, key=lambda x: x.stat.date)
        # whether outputs
        if kwargs.get('output', False):
            # to text, to json, to chart
            [(wrapper.to_text(), wrapper.to_json(), wrapper.to_chart()) for wrapper in wrappers]

            # wrappers
        return wrappers

    def learn(self, solver, env, kwargs):
        # only train the DL/RL optimizer with training mode
        print("训练")
        if not (isinstance(solver.optimizer, GNPOptimizer) and kwargs['train']): return solver
        # supervised training mode
        if kwargs.get('train', True):
            _start_time = time.perf_counter()

            idx = list(range(env.data.n_samples))
            experience = []
            # batch of arguments
            sl_solver = Solver(MISGurobiOptimizer(env.data.positions.shape[-1]))
            args = [(sl_solver, copy(env), self.cfg, k, kwargs) for k in idx]
            # multi-thread training
            for arg in args:
                experience.append(interact(arg)[0])

            dataSet = []
            for one_day_experience in experience:
                for graph_data in one_day_experience:
                    G, label,val = graph_data
                    pref, mask, adj_matrix, edge_attr = G

                    num_nodes = len(pref)
                    if num_nodes == 1:
                        # 跳过节点数为1的图
                        continue
                    # 构造图神经网络数据集
                    data = construct_graph_data(pref, adj_matrix, edge_attr,label,val)
                    dataSet.append(data)

            batch_size, epochs, path = 32, kwargs['epochs'], kwargs['path']

            # step 4.2: train model
            solver.optimizer.Policy.train(dataSet, epochs=epochs, batch_size=batch_size, path=path)
            # 保存经验到Pickle文件
            experience_file = 'experience.pkl'
            with open(experience_file, 'wb') as file:
                pickle.dump(experience, file)

            print(f'Experience data saved to {experience_file}')

        return

    # pickup the model with the best performance on validation set
    def validate(self, solver, env, kwargs):
        # only DL/RL optimizer needs the validation mode
        print("验证")
        if not (isinstance(solver.optimizer, GNROptimizer)) or not kwargs.get('validate', False):  return
        return


# parallel collecting experience
def interact(args):
    # unzip the arguments
    solver, env, cfg, batch_index, kwargs = args

    # unzip the alloc
    dispatch, _start_time_, data = env.reset(cfg.taskConf.Input, batch_index)
    # statistic on the dispatch
    wrapper = DispatchWrapper(data, dispatch, cfg.taskConf.Input,
                              cfg.taskConf.Input.data.plannings[batch_index][-15:-5])
    experience = solver(env, wrapper.dispatch, kwargs)
    wrapper.stat.usage, wrapper.stat.elapsed = env.reward(wrapper.dispatch), time.perf_counter() - _start_time_
    print(f'{wrapper}')
    return experience,wrapper

def average_neighbor_degree(adj):
    degrees = adj.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_neigh_deg = np.where(degrees > 0, (adj @ degrees) / degrees, 0)
    return avg_neigh_deg




@hydra.main(version_base=None, config_path='config', config_name='conf.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    # solver
    solver = hydra.utils.instantiate(cfg.solver)
    # activate the environment
    env = hydra.utils.instantiate(cfg.env)

    n_positions = env.data.positions.shape[-1]

    # initialize the optimizer
    if cfg.hyperparameters['train']:
        solver.optimizer = solver.optimizer(n_positions=n_positions)
    else:
        solver.optimizer = solver.optimizer(n_positions=n_positions,
                                            pretrained=os.path.join(cfg.taskConf.Input.stat.model, '400.pt'))
        # solver.optimizer = solver.optimizer(n_positions=n_positions)
    # training
    workspace.learn(solver, env, dict(cfg.hyperparameters,path = cfg.taskConf.Input.stat.model))

    # validation
    workspace.validate(solver, env, dict(cfg.hyperparameters, train=False, combine=False, use_thread=False))
    # # # inference
    workspace.infer(solver, env, dict(cfg.hyperparameters, train=False, output=True))




if __name__ == '__main__':
    main()
