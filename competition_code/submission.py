"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.straight_speed = 67.0
        self.medium_corner_speed = 40.0
        self.sharp_corner_speed = 25.0

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )




    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        def future_point(offset):
            index = (
                self.current_waypoint_idx + offset
            ) % len(self.maneuverable_waypoints)
            return self.maneuverable_waypoints[
                index
            ].location[:2]
        
        points = [
            future_point(5),
            future_point(15),
            future_point(30),
            future_point(50),
        ]

        headings = []

        for start,end in zip(points[:-1], points[1:]):
            vector_to_waypoint = end - start
            heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])
            headings.append(heading_to_waypoint)            

        turn_angles = [
            abs(normalize_rad(second - first))
            for first, second in zip(
                headings[:-1],
                headings[1:],
            )
        ]

        largest_turn = max(turn_angles, default=0.0)
        
        # what speed to go based on the largest turn angle in the next 4 major waypoints
        # if largest_turn < 0.08:
        #     target_speed = self.straight_speed
        # elif largest_turn < 0.25:
        #     target_speed = self.medium_corner_speed
        # else:
        #     target_speed = self.sharp_corner_speed

        # continuous speed control 
        target_speed = float(
            np.interp(
                largest_turn,
                [0.0, 0.10, 0.30, 0.60],
                [
                    self.straight_speed,
                    self.straight_speed,
                    self.medium_corner_speed,
                    self.sharp_corner_speed,
                ],
            )
        )

        # look ahead distance for steering based on the current speed
        steering_lookahead = int(
            np.clip(
                4 + vehicle_velocity_norm * 0.2,
                4,
                12,
            )
        )
        # next waypoint to follow based on the lookahead distance
        target_index = (
            self.current_waypoint_idx
            + steering_lookahead
        ) % len(self.maneuverable_waypoints)
        waypoint_to_follow = self.maneuverable_waypoints[
            target_index
        ]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Steer control to steer the vehicle towards the target waypoint
        if vehicle_velocity_norm > 1e-2:
            steer_control = (
                -8.0
                / np.sqrt(vehicle_velocity_norm)
                * delta_heading
                / np.pi
            )
        else:
            steer_control = -np.sign(delta_heading)
        steer_control = np.clip(
            steer_control,
            -1.0,
            1.0,
        )   

        if abs(steer_control) > 0.25:
            target_speed = min(target_speed, 20.0)
        elif abs(steer_control) > 0.15:
            target_speed = min(target_speed, 25.0)

        # Throttle control to control the vehicle's speed towards the target speed
        speed_error = (
            target_speed - vehicle_velocity_norm
        )
        if speed_error > 0.5:
            throttle_control = np.clip(
                0.12 + 0.07 * speed_error,
                0.0,
                1.0,
            )
            brake_control = 0.0
        elif speed_error < -0.5:
            throttle_control = 0.0
            brake_control = np.clip(
                0.10 * -speed_error,
                0.0,
                1.0,
            )
        else:
            # Small throttle helps counter vehicle drag
            throttle_control = (
                0.12 if speed_error >= 0 else 0.0
            )
            brake_control = 0.0


        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": brake_control,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control





        # ROAR SAMPLE CODE FOR REFERENCE ONLY
        #  # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
        # waypoint_to_follow = self.maneuverable_waypoints[(self.current_waypoint_idx + 3) % len(self.maneuverable_waypoints)]

        # # Calculate delta vector towards the target waypoint
        # vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        # heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # # Calculate delta angle towards the target waypoint
        # delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # # Proportional controller to steer the vehicle towards the target waypoint
        # steer_control = (
        #     -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        # ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        # steer_control = np.clip(steer_control, -1.0, 1.0)

        # # Proportional controller to control the vehicle's speed towards 40 m/s
        # throttle_control = 0.05 * (20 - vehicle_velocity_norm)