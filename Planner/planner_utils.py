import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from common_utils import *
from .refinement import RefinementPlanner
from .smoother import MotionNonlinearSmoother
from .occupancy_adapter import occupancy_adpter

from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states


class TrajectoryPlanner:
    def __init__(self, device='cpu'):
        self.N = int(T/DT)
        self.ts = DT
        self._device = device
        self.planner = RefinementPlanner(device)
    
    def plan(self, ego_state, ego_state_transformed, neighbors_state_transformed, 
             predictions, plan, scores, ref_path, observation):
        # Get the plan from the prediction model
        plan = plan[0].cpu().numpy()

        # Get the plan in the reference path
        if ref_path is not None:
            distance_to_ref = scipy.spatial.distance.cdist(plan[:, :2], ref_path[:, :2])
            i = np.argmin(distance_to_ref, axis=1)
            plan = ref_path[i, :3]
            s = np.concatenate([[0], i]) * 0.1
            speed = np.diff(s) / DT
        else:
            speed = np.diff(plan[:, :2], axis=0) / DT
            speed = np.linalg.norm(speed, axis=-1)
            speed = np.concatenate([speed, [speed[-1]]])
            
        # Refine planning
        if ref_path is None:
            pass
        else:
            occupancy = occupancy_adpter(predictions[0], scores[0, 1:], neighbors_state_transformed[0], ref_path)
            ego_plan_ds = torch.from_numpy(speed).float().unsqueeze(0).to(self._device)
            ego_plan_s = torch.from_numpy(s).float().unsqueeze(0).to(self._device)
            ego_state_transformed = ego_state_transformed.to(self._device)
            ref_path = torch.from_numpy(ref_path).unsqueeze(0).to(self._device)
            occupancy = torch.from_numpy(occupancy).unsqueeze(0).to(self._device)

            s, speed = self.planner.plan(ego_state_transformed, ego_plan_ds, ego_plan_s, occupancy, ref_path)
            s = s.squeeze(0).cpu().numpy()
            speed = speed.squeeze(0).cpu().numpy()

            # Convert to Cartesian trajectory
            ref_path = ref_path.squeeze(0).cpu().numpy()
            i = (s * 10).astype(np.int32).clip(0, len(ref_path)-1)
            plan = ref_path[i, :3]

        return plan
    
    @staticmethod
    def transform_to_Cartesian_path(path, ref_path):
        frenet_idx = np.array(path[:, 0] * 10, dtype=np.int32)
        frenet_idx = np.clip(frenet_idx, 0, len(ref_path)-1)
        ref_points = ref_path[frenet_idx]
        l = path[frenet_idx, 1]

        cartesian_x = ref_points[:, 0] - l * np.sin(ref_points[:, 2])
        cartesian_y = ref_points[:, 1] + l * np.cos(ref_points[:, 2])
        cartesian_path = np.column_stack([cartesian_x, cartesian_y])

        return cartesian_path


def annotate_occupancy(occupancy, ego_path, red_light_lane):
    ego_path_red_light = scipy.spatial.distance.cdist(ego_path[:, :2], red_light_lane)

    if len(red_light_lane) < 80:
        pass
    else:
        occupancy[np.any(ego_path_red_light < 0.5, axis=-1)] = 1

    return occupancy


def annotate_speed(ref_path, speed_limit):
    speed = np.ones(len(ref_path)) * speed_limit
    
    # get the turning point
    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

    # set speed limit to 3 m/s for turning
    if turning_idx > 0:
        speed[turning_idx:] = 3

    return speed[:, None]


def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi


def transform_to_ego_frame(path, ego_state):
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
    path_x, path_y = path[:, 0], path[:, 1]
    ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
    ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
    ego_path = np.stack([ego_path_x, ego_path_y], axis=-1)

    return ego_path
