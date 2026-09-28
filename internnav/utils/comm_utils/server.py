#!/usr/bin/env python
import base64
import multiprocessing
import pickle
from typing import Dict
import importlib.util
import sys

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status

from internnav.agent.base import Agent
from internnav.configs.agent import InitRequest, ResetRequest, StepRequest


class AgentServer:
    """
    Server class for Agent service.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = FastAPI(title='Agent Service')
        self.agent_instances: Dict[str, Agent] = {}
        self._router = APIRouter(prefix='/agent')
        self._register_routes()
        self.app.include_router(self._router)

    def _register_routes(self):
        route_config = [
            ('/init', self.init_agent, ['POST'], status.HTTP_201_CREATED),
            ('/{agent_name}/step', self.step_agent, ['POST'], None),
            ('/{agent_name}/reset', self.reset_agent, ['POST'], None),
            # TODO: Add stop server route
        ]

        for path, handler, methods, status_code in route_config:
            self._router.add_api_route(
                path=path,
                endpoint=handler,
                methods=methods,
                status_code=status_code,
            )

    async def init_agent(self, request: InitRequest):
        agent_config = request.agent_config
        agent = Agent.init(agent_config)
        agent_name = agent_config.model_name
        self.agent_instances[agent_name] = agent
        return {'status': 'success', 'agent_name': agent_name}

    async def step_agent(self, agent_name: str, request: StepRequest):
        self._validate_agent_exists(agent_name)
        agent = self.agent_instances[agent_name]

        def transfer(obs):
            obs = base64.b64decode(obs)
            obs = pickle.loads(obs)
            return obs

        obs = transfer(request.observation)
        action = agent.step(obs)
        return {'action': action}

    async def reset_agent(self, agent_name: str, request: ResetRequest):
        self._validate_agent_exists(agent_name)
        # Forward both reset_index and episode_ids to the agent
        reset_index = getattr(request, 'reset_index', None)
        episode_ids = getattr(request, 'episode_ids', None)
        self.agent_instances[agent_name].reset(reset_index, episode_ids=episode_ids)
        return {'status': 'success'}

    def _validate_agent_exists(self, agent_name: str):
        if agent_name not in self.agent_instances:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Agent not found')

    def run(self, reload=False):
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            reload=reload,
            reload_dirs=['./internnav/agent/', './internnav/model/'],
        )


def start_server(host='localhost', port=8087, dist=False):
    """
    start a server in the backgrouond process

    Args:
        host
        port

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_run_server if not dist else _run_server_dist, args=(host, port))
    p.daemon = True
    p.start()
    print(f"Server started on {host}:{port} (pid={p.pid})")
    return p


def _run_server_dist(host='localhost', port=8087):
    import torch

    from internnav.utils.dist import get_rank

    device_idx = get_rank()
    torch.cuda.set_device(device_idx)
    print(f"Server using GPU {device_idx}")
    server = AgentServer(host, port)
    server.run()


def _run_server(host='localhost', port=8087):
    server = AgentServer(host, port)
    server.run()

def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location('eval_config_module', config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['eval_config_module'] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, 'eval_cfg')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/eval/configs/h1_cma_cfg.py',
        help='eval config file path, e.g. scripts/eval/configs/h1_cma_cfg.py',
    )
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    eval_cfg = load_eval_cfg(args.config)
    args.port = eval_cfg.agent.server_port

    server = AgentServer(args.host, args.port)
    server.run(args.reload)

