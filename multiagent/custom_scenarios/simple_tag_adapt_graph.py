import numpy as np
from typing import Optional, Tuple, List
from numpy import ndarray as arr
from scipy import sparse
import argparse
from multiagent.core import World, Agent, Landmark, Entity, Action
from multiagent.scenario import BaseScenario

entity_mapping = {"agent": 0, "landmark": 1}

class Scenario(BaseScenario):
    def make_world(self, args: argparse.Namespace) -> World:
        world = World()
        world.current_time_step = 0
        # set any world properties first
        world.dim_c = 2
        world.world_length = getattr(args, "episode_length", 25)
        num_good_agents = getattr(args, "num_good_agents", 1)
        num_adversaries = getattr(args, "num_adversaries", 3)
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0 # No blocks in ADAPT setup
        
        # Automatic scaling env according to agent number (logarithmic)
        world.arena_size = max(1.0, np.log(max(2, num_agents)) / np.log(6.0))
        
        # Graph properties
        world.graph_mode = True
        world.cache_dists = True
        if hasattr(args, "graph_feat_type"):
            world.graph_feat_type = args.graph_feat_type
        else:
            world.graph_feat_type = "global"
            
        if not hasattr(args, "max_edge_dist"):
            self.max_edge_dist = 1
        else:
            self.max_edge_dist = args.max_edge_dist

        # add agents
        global_id = 0
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.global_id = global_id
            global_id += 1
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            
            # ADAPT rule-based prey
            if not agent.adversary:
                agent.action_callback = self.prey_policy
                world.scripted_agents.append(agent)
                
        # add landmarks (none)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        
        # Custom step logic to update global state after physics
        original_step = world.step
        def custom_step():
            original_step()
            # update caught_matrix here!
            for adv_idx, adv in enumerate(self.adversaries(world)):
                for ag_idx, ag in enumerate(self.good_agents(world)):
                    if self.is_collision(ag, adv):
                        if not self.caught_matrix[adv_idx, ag_idx]:
                            self.caught_matrix[adv_idx, ag_idx] = True
                            self.newly_caught[adv_idx, ag_idx] = True
            if np.all(np.any(self.caught_matrix, axis=0)):
                self.game_winning_condition_met = True
                
        world.step = custom_step
            
        # Initial states
        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        world.current_time_step = 0
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        
        # ADAPT specific state tracking
        num_adversaries = len(self.adversaries(world))
        num_good_agents = len(self.good_agents(world))
        self.caught_matrix = np.zeros((num_adversaries, num_good_agents), dtype=bool)
        self.newly_caught = np.zeros((num_adversaries, num_good_agents), dtype=bool)
        self.game_winning_condition_met = False
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.arena_size, +world.arena_size, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        if hasattr(world, 'calculate_distances'):
            world.calculate_distances()
        self.update_graph(world)

    def prey_policy(self, agent, world):
        # Rule-based policy for prey: 
        # If inside [-arena_size, arena_size], choose a random action.
        # If outside, move towards the center (inside) as quickly as possible.
        action = Action()
        action.u = np.zeros(world.dim_p)
        action.c = np.zeros(world.dim_c)
        
        pos = agent.state.p_pos
        
        # Check if out of bounds
        if pos[0] < -world.arena_size or pos[0] > world.arena_size or pos[1] < -world.arena_size or pos[1] > world.arena_size:
            # Find the axis and direction that is most out of bounds
            if abs(pos[0]) > abs(pos[1]):
                if pos[0] > 0:
                    dir = 1 # move left (-x)
                else:
                    dir = 2 # move right (+x)
            else:
                if pos[1] > 0:
                    dir = 3 # move down (-y)
                else:
                    dir = 4 # move up (+y)
        else:
            dir = np.random.randint(0, 5) # 0: stay, 1: left, 2: right, 3: down, 4: up
            
        if dir == 1: action.u[0] -= 1.0
        elif dir == 2: action.u[0] += 1.0
        elif dir == 3: action.u[1] -= 1.0
        elif dir == 4: action.u[1] += 1.0
            
        sensitivity = 5.0
        if agent.accel is not None:
            sensitivity = agent.accel
        action.u *= sensitivity
        
        return action

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Only predators are trained using RL, so we only return reward for adversaries
        if not agent.adversary:
            return 0.0
            
        rew = 0.0
        adversaries = self.adversaries(world)
        adv_idx = adversaries.index(agent)
        
        # Check newly caught for this predator
        for ag_idx in range(self.newly_caught.shape[1]):
            if self.newly_caught[adv_idx, ag_idx]:
                rew += 10.0 # Reward for first-time capture
                self.newly_caught[adv_idx, ag_idx] = False # Clear flag after rewarding
                    
        # Check if game-winning condition is met
        if self.game_winning_condition_met:
            rew += 20.0
            
        return rew

    def info_callback(self, agent: Agent, world: World):
        return {'is_success': self.game_winning_condition_met}

    def observation(self, agent, world):
        # In graph-based MARL, relational features are handled by the GNN.
        # The standard observation should only contain fixed-size ego features
        # to allow zero-shot scaling to different numbers of agents.
        is_prey = 1.0 if not agent.adversary else 0.0
        is_predator = 1.0 if agent.adversary else 0.0
        
        # Self features (dim = 6: vel(2) + pos(2) + is_prey(1) + is_predator(1))
        return np.concatenate([agent.state.p_vel, agent.state.p_pos, [is_prey, is_predator]])

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.global_id])

    def graph_observation(self, agent: Agent, world: World) -> Tuple[arr, arr]:
        node_obs = []
        if world.graph_feat_type == "global":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_global(entity, world)
                node_obs.append(node_obs_i)
        elif world.graph_feat_type == "relative":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_relative(agent, entity, world)
                node_obs.append(node_obs_i)

        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag
        return node_obs, adj

    def update_graph(self, world: World):
        if not hasattr(world, 'cached_dist_mag'):
            world.calculate_distances()
        dists = world.cached_dist_mag
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        if world.graph_feat_type == "global":
            world.edge_weight = dists[row, col]
        elif world.graph_feat_type == "relative":
            world.edge_weight = dists[row, col]

    def _get_entity_feat_global(self, entity: Entity, world: World) -> arr:
        pos = entity.state.p_pos
        vel = entity.state.p_vel
        goal_pos = np.zeros_like(pos)
        
        is_prey = 1.0 if (hasattr(entity, 'adversary') and not entity.adversary) else 0.0
        is_predator = 1.0 if (hasattr(entity, 'adversary') and entity.adversary) else 0.0
        entity_type = entity_mapping.get("agent" if isinstance(entity, Agent) else "landmark", 0)

        return np.hstack([vel, pos, goal_pos, is_prey, is_predator, entity_type])

    def _get_entity_feat_relative(
        self, agent: Agent, entity: Entity, world: World
    ) -> arr:
        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel
        entity_pos = entity.state.p_pos
        entity_vel = entity.state.p_vel
        rel_pos = entity_pos - agent_pos
        rel_vel = entity_vel - agent_vel
        
        rel_goal_pos = np.zeros_like(rel_pos)
        is_prey = 1.0 if (hasattr(entity, 'adversary') and not entity.adversary) else 0.0
        is_predator = 1.0 if (hasattr(entity, 'adversary') and entity.adversary) else 0.0
        entity_type = entity_mapping.get("agent" if isinstance(entity, Agent) else "landmark", 0)

        return np.hstack([rel_vel, rel_pos, rel_goal_pos, is_prey, is_predator, entity_type])
