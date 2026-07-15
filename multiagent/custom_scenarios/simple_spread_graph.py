import numpy as np
from typing import Optional, Tuple, List
from numpy import ndarray as arr
from scipy import sparse
import argparse
from multiagent.core import World, Agent, Landmark, Entity
from multiagent.scenario import BaseScenario

entity_mapping = {"agent": 0, "landmark": 1}

class Scenario(BaseScenario):
    def make_world(self, args: argparse.Namespace) -> World:
        world = World()
        world.world_length = args.episode_length
        world.current_time_step = 0
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks 
        world.collaborative = True
        
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
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.global_id = global_id
            global_id += 1
            agent.size = 0.08
            
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.global_id = global_id
            global_id += 1
            
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World):
        world.current_time_step = 0
        # random properties for agents
        world.assign_agent_colors()
        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_vel = np.zeros(world.dim_p)
            # Try up to 100 times to find a position not too close to other landmarks
            best_pos = None
            max_min_dist = -1
            for _ in range(100):
                pos = 1 * np.random.uniform(-1, +1, world.dim_p)
                if i == 0:
                    landmark.state.p_pos = pos
                    break
                dists = [np.linalg.norm(pos - world.landmarks[j].state.p_pos) for j in range(i)]
                min_dist = min(dists)
                if min_dist > 0.6:  # Minimum distance threshold
                    landmark.state.p_pos = pos
                    break
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_pos = pos
            else:
                # Fallback if we can't find a spaced-out position after 100 attempts
                landmark.state.p_pos = best_pos if best_pos is not None else pos
                
        # Update graph structures at reset
        world.calculate_distances()
        self.update_graph(world)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1

        # Add a boundary penalty to prevent agents from running out of bounds
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
            
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def observation(self, agent, world):
        # In graph-based MARL, relational features are handled by the GNN.
        # The standard observation should only contain fixed-size ego features
        # to allow zero-shot scaling to different numbers of agents.
        return np.concatenate([agent.state.p_vel, agent.state.p_pos])

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.global_id])

    def info_callback(self, agent: Agent, world: World):
        # Calculate minimum distance to any landmark for this agent
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        min_dist = min(dists) if dists else 0.0
        
        # Calculate collisions
        agent_collisions = 0
        for a in world.agents:
            if a is not agent and self.is_collision(a, agent):
                agent_collisions += 1
                
        # Return expected keys to prevent IndexError in runner's get_collisions and get_fraction_episodes
        return {
            "Dist_to_goal": min_dist,
            "Time_req_to_goal": 0.0,  # Dummy value since spread doesn't track this
            "Num_agent_collisions": float(agent_collisions),
            "Num_obst_collisions": 0.0,
            "Min_time_to_goal": 0.0
        }

    def graph_observation(self, agent: Agent, world: World) -> Tuple[arr, arr]:
        num_entities = len(world.entities)
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
        if "agent" in entity.name:
            # simple spread doesn't assign specific goals to agents, so we use a dummy goal
            goal_pos = np.zeros_like(pos)
            entity_type = entity_mapping["agent"]
        elif "landmark" in entity.name:
            goal_pos = pos
            entity_type = entity_mapping["landmark"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([vel, pos, goal_pos, entity_type])

    def _get_entity_feat_relative(
        self, agent: Agent, entity: Entity, world: World
    ) -> arr:
        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel
        entity_pos = entity.state.p_pos
        entity_vel = entity.state.p_vel
        rel_pos = entity_pos - agent_pos
        rel_vel = entity_vel - agent_vel
        
        if "agent" in entity.name:
            rel_goal_pos = np.zeros_like(rel_pos)
            entity_type = entity_mapping["agent"]
        elif "landmark" in entity.name:
            rel_goal_pos = rel_pos
            entity_type = entity_mapping["landmark"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type])
