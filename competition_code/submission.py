"""
# Competition instructions:
# Please do not change anything else but fill out the to-do sections.

"""

from typing import List

import numpy as np
import roar_py_interface


def normalize_rad(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.official_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()

        self.maximum_straight_speed = 89.4
        self.minimum_corner_speed = 17.0
        # Keep wide/flowing corners above roughly 80 mph. Tight corners
        # remain curvature-limited by minimum_corner_speed.
        self.minimum_flowing_turn_speed = 35.8
        self.maximum_flowing_turn_speed = 40.2
        self.straight_setup_offset = 1.75
        self.maximum_lateral_offset = 2.25
        self.maximum_deceleration = 7.5
        self.previous_steer = 0.0
        self.steer_is_unwinding = False
        self.previous_speed = 0.0
        self.brake_ticks_remaining = 0
        self.brake_release_ticks = 0

        self.centerline = np.array(
            [waypoint.location[:2] for waypoint in self.official_waypoints],
            dtype=float,
        )
        self.waypoint_count = len(self.centerline)

        self.corner_segments = [
            (127, 138, 162),
            (238, 265, 293),
            (337, 342, 347),
            (434, 444, 449),
            (481, 509, 549),
            (675, 723, 746),
            (851, 874, 895),
            (956, 967, 973),
            (1354, 1371, 1388),
            (1409, 1468, 1486),
            (1925, 1963, 1991),
            (2012, 2019, 2037),
            (2065, 2073, 2080),
            (2697, 2752, 2774),
        ]
        self.hairpin_segments = [
            (1354, 1371, 1388),
            (2697, 2752, 2774),
        ]
        self.opening_full_throttle_end = self.corner_segments[3][0]
        self.opening_right_offset = -self.straight_setup_offset
        self.middle_full_throttle_start = 900
        self.middle_full_throttle_end = 1150
        self.middle_left_offset = self.maximum_lateral_offset
        self.curve_ten_full_throttle_start = 1470
        self.curve_ten_full_throttle_end = 1850
        self.back_straight_path_start = 2000
        self.back_straight_full_throttle_start = 2100
        self.back_straight_end = 2580
        self.back_straight_right_offset = -self.maximum_lateral_offset

        self.racing_line = self._build_racing_line()
        self.curvature_profile = self._build_curvature_profile(self.racing_line)
        self.speed_profile = self._build_speed_profile(
            self.racing_line,
            self.curvature_profile,
        )

        distances = np.linalg.norm(
            self.racing_line - vehicle_location[:2],
            axis=1,
        )
        self.current_waypoint_idx = int(np.argmin(distances))

        self.tick_count = 0
        self.seconds_per_tick = 0.05
        self.total_waypoints_passed = 0
        self.previous_waypoint_idx = self.current_waypoint_idx
        self.checkpoint_interval = 50
        self.next_checkpoint = 50
        self.previous_checkpoint_time = 0.0

    def _build_racing_line(self) -> np.ndarray:
        previous_points = np.roll(self.centerline, 1, axis=0)
        next_points = np.roll(self.centerline, -1, axis=0)

        tangents = next_points - previous_points
        tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(tangent_lengths, 1e-9)

        left_normals = np.column_stack(
            (-tangents[:, 1], tangents[:, 0])
        )

        offset_sum = np.zeros(self.waypoint_count, dtype=float)
        offset_weight = np.zeros(self.waypoint_count, dtype=float)

        def signed_curvature_at(index: int, span: int = 6) -> float:
            point_a = self.centerline[(index - span) % self.waypoint_count]
            point_b = self.centerline[index % self.waypoint_count]
            point_c = self.centerline[(index + span) % self.waypoint_count]
            distance_ab = np.linalg.norm(point_b - point_a)
            distance_bc = np.linalg.norm(point_c - point_b)
            distance_ca = np.linalg.norm(point_c - point_a)
            denominator = distance_ab * distance_bc * distance_ca
            if denominator < 1e-6:
                return 0.0
            cross = (
                (point_b[0] - point_a[0])
                * (point_c[1] - point_a[1])
                - (point_b[1] - point_a[1])
                * (point_c[0] - point_a[0])
            )
            return float(2.0 * cross / denominator)

        for turn_number, (entry, apex, exit_index) in enumerate(
            self.corner_segments,
            start=1,
        ):
            segment_indices = list(range(entry, exit_index + 1))
            segment_curvatures = np.array(
                [signed_curvature_at(index) for index in segment_indices],
                dtype=float,
            )
            dominant_index = int(np.argmax(np.abs(segment_curvatures)))
            dominant_curvature = float(segment_curvatures[dominant_index])
            turn_direction = 1.0 if dominant_curvature >= 0.0 else -1.0

            meaningful_left = np.any(segment_curvatures > 0.002)
            meaningful_right = np.any(segment_curvatures < -0.002)
            is_s_turn = (
                turn_number == 5
                or (meaningful_left and meaningful_right)
            )

            heading_span = 10
            incoming = (
                self.centerline[entry]
                - self.centerline[
                    (entry - heading_span) % self.waypoint_count
                ]
            )
            outgoing = (
                self.centerline[
                    (exit_index + heading_span) % self.waypoint_count
                ]
                - self.centerline[exit_index]
            )
            heading_denominator = max(
                np.linalg.norm(incoming) * np.linalg.norm(outgoing),
                1e-9,
            )
            turn_angle_degrees = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(incoming, outgoing)
                            / heading_denominator,
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            is_wide_turn = not is_s_turn and turn_angle_degrees < 40.0

            is_hairpin = (entry, apex, exit_index) in self.hairpin_segments
            setup_start = entry - (180 if is_hairpin else 150)
            approach_start = entry - (100 if is_hairpin else 80)
            turn_in = entry
            late_apex = min(
                apex + (8 if is_hairpin else 3),
                exit_index - 1,
            )
            departure_end = exit_index + (90 if is_hairpin else 70)

            inside_offset = turn_direction * self.maximum_lateral_offset
            outside_offset = -inside_offset
            first_half_count = max(
                (exit_index - entry) // 2 + 1,
                1,
            )
            first_half = segment_curvatures[:first_half_count]
            first_dominant = float(
                first_half[int(np.argmax(np.abs(first_half)))]
            )
            first_direction = 1.0 if first_dominant >= 0.0 else -1.0
            if turn_number == 5:
                first_direction = 1.0
            first_inside = (
                first_direction * self.maximum_lateral_offset
            )

            if is_s_turn:
                straight_offset = (
                    first_direction * self.straight_setup_offset
                )
            else:
                straight_offset = (
                    -turn_direction * self.straight_setup_offset
                )

            if turn_number == 10:
                # Use one moderate inside apex instead of either staying
                # centered or switching across both sides. This supplies a
                # clear turning target without aiming toward the outer wall.
                curve_ten_inside = 1.50 * turn_direction
                curve_ten_outside = -0.50 * turn_direction
                controls = [
                    (setup_start, 0.0),
                    (approach_start, curve_ten_outside),
                    (turn_in, curve_ten_outside),
                    (apex, curve_ten_inside),
                    (exit_index, 0.0),
                    (departure_end, 0.0),
                ]
            elif turn_number == 11:
                # Following-corner setup windows overlap this entry, so force
                # a clear right-side approach before moving toward the apex.
                curve_eleven_right_approach = -2.0
                controls = [
                    (setup_start, 0.0),
                    (
                        approach_start,
                        -self.straight_setup_offset,
                    ),
                    (turn_in, curve_eleven_right_approach),
                    (late_apex, 0.80 * inside_offset),
                    (exit_index, 0.20 * outside_offset),
                    (departure_end, 0.0),
                ]
            elif turn_number in (6, 7):
                # Both bends can use more of the right side. Begin the sharp
                # transition ten waypoints earlier and hold the right apex.
                original_turn_in = entry + max(
                    int(0.55 * (apex - entry)), 1
                )
                turn_in_advance = 18 if turn_number == 7 else 10
                sharp_turn_in = max(
                    entry,
                    original_turn_in - turn_in_advance,
                )
                apex_hold_end = min(apex + 10, exit_index - 1)
                right_apex_offset = (
                    -2.25 if turn_number == 7 else -2.5
                )
                left_setup_offset = self.maximum_lateral_offset
                controls = [
                    (setup_start, 0.0),
                    (approach_start, self.straight_setup_offset),
                    (turn_in, left_setup_offset),
                    (sharp_turn_in, left_setup_offset),
                    (apex, right_apex_offset),
                    (apex_hold_end, right_apex_offset),
                    (exit_index, 0.70 * right_apex_offset),
                    (departure_end, 0.0),
                ]
            elif is_s_turn:
                # Follow the inside of the first bend, cross the center at
                # the midpoint, then follow the inside of the opposite bend.
                midpoint = (entry + exit_index) // 2
                first_apex = entry + max((midpoint - entry) // 2, 1)
                second_apex = midpoint + max(
                    (exit_index - midpoint) // 2,
                    1,
                )
                controls = [
                    (setup_start, 0.0),
                    (approach_start, straight_offset),
                    (turn_in, first_inside),
                    (first_apex, first_inside),
                    (midpoint, 0.0),
                    (second_apex, -first_inside),
                    (exit_index, -first_inside),
                    (departure_end, 0.0),
                ]
            elif is_wide_turn:
                # Broad sweepers favor the shorter inside path and need very
                # little steering correction, so remain inside throughout.
                controls = [
                    (setup_start, 0.0),
                    (approach_start, straight_offset),
                    (turn_in, inside_offset),
                    (late_apex, inside_offset),
                    (exit_index, inside_offset),
                    (departure_end, 0.0),
                ]
            else:
                # Tighter isolated corners use outside-inside-outside.
                controls = [
                    (setup_start, 0.0),
                    (approach_start, straight_offset),
                    (turn_in, outside_offset),
                    (late_apex, inside_offset),
                    (exit_index, outside_offset),
                    (departure_end, 0.0),
                ]

            for (start, start_offset), (end, end_offset) in zip(
                controls[:-1],
                controls[1:],
            ):
                segment_length = max(end - start, 1)
                control_weight = 3.0 if turn_number == 11 else 1.0

                for unwrapped_index in range(start, end + 1):
                    progress = (unwrapped_index - start) / segment_length
                    blend = smoothstep(progress)
                    offset = (
                        start_offset
                        + blend * (end_offset - start_offset)
                    )
                    index = unwrapped_index % self.waypoint_count
                    offset_sum[index] += offset * control_weight
                    offset_weight[index] += control_weight

        lateral_offsets = np.divide(
            offset_sum,
            offset_weight,
            out=np.zeros(self.waypoint_count, dtype=float),
            where=offset_weight > 0.0,
        )

        # Limit how quickly the requested path can move laterally. Start from
        # a known straight so the circular final corner is handled cleanly.
        def maximum_offset_change(index: int) -> float:
            if (
                481 <= index <= 549
                or 675 <= index <= 746
                or 851 <= index <= 895
            ):
                return 0.25
            return 0.05

        ordered_indices = (
            1000 + np.arange(self.waypoint_count)
        ) % self.waypoint_count

        for index in ordered_indices[1:]:
            previous_index = (index - 1) % self.waypoint_count
            change_limit = maximum_offset_change(int(index))
            lateral_offsets[index] = np.clip(
                lateral_offsets[index],
                lateral_offsets[previous_index] - change_limit,
                lateral_offsets[previous_index] + change_limit,
            )

        for index in ordered_indices[-2::-1]:
            next_index = (index + 1) % self.waypoint_count
            change_limit = maximum_offset_change(int(index))
            lateral_offsets[index] = np.clip(
                lateral_offsets[index],
                lateral_offsets[next_index] - change_limit,
                lateral_offsets[next_index] + change_limit,
            )

        # Additional smoothing removes small curvature spikes introduced by
        # overlapping corner transition windows.
        for _ in range(12):
            lateral_offsets = (
                np.roll(lateral_offsets, 1)
                + 2.0 * lateral_offsets
                + np.roll(lateral_offsets, -1)
            ) / 4.0

        # Opening sector: hold the right side through the run to Curve 4.
        # Blend both boundaries to avoid injecting a curvature spike into
        # the generated path.
        opening_end = self.opening_full_throttle_end
        boundary_blend_length = 30
        opening_start_offset = float(lateral_offsets[-1])
        opening_end_offset = float(lateral_offsets[opening_end])

        for index in range(opening_end):
            if index < boundary_blend_length:
                blend = smoothstep(index / boundary_blend_length)
                lateral_offsets[index] = (
                    opening_start_offset
                    + blend
                    * (self.opening_right_offset - opening_start_offset)
                )
            elif index >= opening_end - boundary_blend_length:
                progress = (
                    index - (opening_end - boundary_blend_length)
                ) / boundary_blend_length
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    self.opening_right_offset
                    + blend
                    * (opening_end_offset - self.opening_right_offset)
                )
            else:
                lateral_offsets[index] = self.opening_right_offset

        # Curve 5 must commit left before its first bend. Apply this after
        # global smoothing so overlapping corner windows cannot pull the
        # requested S-line back toward the center.
        curve_five_entry, _curve_five_apex, curve_five_exit = (
            self.corner_segments[4]
        )
        curve_five_midpoint = (
            curve_five_entry + curve_five_exit
        ) // 2
        curve_five_first_apex = (
            curve_five_entry + curve_five_midpoint
        ) // 2
        curve_five_second_apex = (
            curve_five_midpoint + curve_five_exit
        ) // 2
        curve_five_start = curve_five_entry - 40
        curve_five_end = curve_five_exit + 40
        curve_five_start_offset = float(
            lateral_offsets[curve_five_start]
        )
        curve_five_end_offset = float(
            lateral_offsets[curve_five_end]
        )

        for index in range(curve_five_start, curve_five_end + 1):
            if index < curve_five_entry:
                progress = (
                    index - curve_five_start
                ) / (curve_five_entry - curve_five_start)
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    curve_five_start_offset
                    + blend
                    * (
                        self.maximum_lateral_offset
                        - curve_five_start_offset
                    )
                )
            elif index <= curve_five_first_apex:
                lateral_offsets[index] = self.maximum_lateral_offset
            elif index <= curve_five_midpoint:
                progress = (
                    index - curve_five_first_apex
                ) / max(
                    curve_five_midpoint - curve_five_first_apex,
                    1,
                )
                lateral_offsets[index] = (
                    self.maximum_lateral_offset
                    * (1.0 - smoothstep(progress))
                )
            elif index <= curve_five_second_apex:
                progress = (
                    index - curve_five_midpoint
                ) / max(
                    curve_five_second_apex - curve_five_midpoint,
                    1,
                )
                lateral_offsets[index] = (
                    -self.maximum_lateral_offset
                    * smoothstep(progress)
                )
            elif index <= curve_five_exit:
                lateral_offsets[index] = -self.maximum_lateral_offset
            else:
                progress = (
                    index - curve_five_exit
                ) / max(curve_five_end - curve_five_exit, 1)
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    -self.maximum_lateral_offset
                    + blend
                    * (
                        curve_five_end_offset
                        + self.maximum_lateral_offset
                    )
                )

        # Full-throttle middle sector: remain on the left from waypoint
        # 900 through 1150, with smooth entry and exit transitions.
        middle_start = self.middle_full_throttle_start
        middle_end = self.middle_full_throttle_end
        middle_start_offset = float(lateral_offsets[middle_start])
        middle_end_offset = float(lateral_offsets[middle_end])

        for index in range(middle_start, middle_end):
            if index < middle_start + boundary_blend_length:
                progress = (index - middle_start) / boundary_blend_length
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    middle_start_offset
                    + blend
                    * (self.middle_left_offset - middle_start_offset)
                )
            elif index >= middle_end - boundary_blend_length:
                progress = (
                    index - (middle_end - boundary_blend_length)
                ) / boundary_blend_length
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    self.middle_left_offset
                    + blend
                    * (middle_end_offset - self.middle_left_offset)
                )
            else:
                lateral_offsets[index] = self.middle_left_offset

        # Stay near the right edge for the long back straight. Begin shaping
        # the line at waypoint 2000, but do not force full throttle until the
        # linked bends are complete at waypoint 2100.
        straight_start = self.back_straight_path_start
        straight_end = self.back_straight_end
        straight_start_offset = float(lateral_offsets[straight_start])
        straight_end_offset = float(lateral_offsets[straight_end])

        for index in range(straight_start, straight_end):
            if index < straight_start + boundary_blend_length:
                progress = (
                    index - straight_start
                ) / boundary_blend_length
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    straight_start_offset
                    + blend
                    * (
                        self.back_straight_right_offset
                        - straight_start_offset
                    )
                )
            elif index >= straight_end - boundary_blend_length:
                progress = (
                    index - (straight_end - boundary_blend_length)
                ) / boundary_blend_length
                blend = smoothstep(progress)
                lateral_offsets[index] = (
                    self.back_straight_right_offset
                    + blend
                    * (
                        straight_end_offset
                        - self.back_straight_right_offset
                    )
                )
            else:
                lateral_offsets[index] = self.back_straight_right_offset

        return self.centerline + left_normals * lateral_offsets[:, None]

    def _build_curvature_profile(self, path: np.ndarray) -> np.ndarray:
        curvature_sets = []

        for span in (4, 8, 16):
            point_a = np.roll(path, span, axis=0)
            point_b = path
            point_c = np.roll(path, -span, axis=0)

            distance_ab = np.linalg.norm(point_b - point_a, axis=1)
            distance_bc = np.linalg.norm(point_c - point_b, axis=1)
            distance_ca = np.linalg.norm(point_c - point_a, axis=1)

            cross = np.abs(
                (point_b[:, 0] - point_a[:, 0])
                * (point_c[:, 1] - point_a[:, 1])
                - (point_b[:, 1] - point_a[:, 1])
                * (point_c[:, 0] - point_a[:, 0])
            )

            denominator = distance_ab * distance_bc * distance_ca
            curvature = np.divide(
                2.0 * cross,
                denominator,
                out=np.zeros(self.waypoint_count, dtype=float),
                where=denominator > 1e-6,
            )
            curvature_sets.append(curvature)

        return np.max(np.array(curvature_sets), axis=0)

    def _build_speed_profile(
        self,
        path: np.ndarray,
        curvature: np.ndarray,
    ) -> np.ndarray:
        speed_profile = np.full(
            self.waypoint_count,
            self.maximum_straight_speed,
            dtype=float,
        )

        effective_friction = np.interp(
            curvature,
            [0.0, 0.006, 0.012, 0.025, 0.050],
            [2.5, 2.4, 2.2, 1.9, 1.7],
        )

        # Ignore tiny curvature noise so real straights retain the full
        # 200 mph target.
        curved = curvature >= 0.002
        speed_profile[curved] = np.sqrt(
            effective_friction[curved] * 9.81 / curvature[curved]
        )
        speed_profile = np.clip(
            speed_profile,
            self.minimum_corner_speed,
            self.maximum_straight_speed,
        )

        # Very wide turns retain the full straight-line target. Medium
        # flowing turns remain between roughly 70 and 90 mph.
        very_wide_turn = (
            (curvature >= 0.002)
            & (curvature <= 0.006)
        )
        speed_profile[very_wide_turn] = self.maximum_straight_speed

        flowing_turn = (
            (curvature > 0.006)
            & (curvature <= 0.018)
        )
        speed_profile[flowing_turn] = np.clip(
            speed_profile[flowing_turn],
            self.minimum_flowing_turn_speed,
            self.maximum_flowing_turn_speed,
        )

        # Curve 10 uses the semi-straights but protects the apex: carry
        # 100 mph initially, brake to 60 mph at waypoint 1450, hold through
        # the apex, then force full throttle from waypoint 1470.
        curve_ten_entry, curve_ten_apex, curve_ten_exit = (
            self.corner_segments[9]
        )
        curve_ten_braking_start = 1430
        curve_ten_slow_point = 1450
        curve_ten_fast_speed = 44.70
        curve_ten_slow_speed = 26.82

        speed_profile[
            curve_ten_entry : curve_ten_braking_start + 1
        ] = curve_ten_fast_speed
        speed_profile[
            curve_ten_braking_start : curve_ten_slow_point + 1
        ] = np.linspace(
            curve_ten_fast_speed,
            curve_ten_slow_speed,
            curve_ten_slow_point - curve_ten_braking_start + 1,
        )
        speed_profile[
            curve_ten_slow_point : self.curve_ten_full_throttle_start
        ] = curve_ten_slow_speed

        # Follow the right wall through the linked bends after waypoint 2000.
        # Raise the target from 90 to 110 mph, then release full throttle once
        # the road becomes straight at waypoint 2100.
        back_turn_start = self.back_straight_path_start
        back_turn_end = 2080
        back_turn_release = self.back_straight_full_throttle_start
        speed_profile[back_turn_start : back_turn_end + 1] = np.linspace(
            40.23,
            49.17,
            back_turn_end - back_turn_start + 1,
        )
        speed_profile[back_turn_end:back_turn_release] = 49.17

        # Turn 6 can retain the 80 mph floor. Turn 7 needs a lower entry and
        # apex speed after its outward-wall understeer, then accelerates from
        # the apex through the normal corner-exit target logic.
        turn_six_entry, _turn_six_apex, turn_six_exit = (
            self.corner_segments[5]
        )
        speed_profile[turn_six_entry : turn_six_exit + 1] = np.maximum(
            speed_profile[turn_six_entry : turn_six_exit + 1],
            35.8,
        )

        turn_seven_entry, _turn_seven_apex, turn_seven_exit = (
            self.corner_segments[6]
        )
        speed_profile[turn_seven_entry : turn_seven_exit + 1] = np.minimum(
            speed_profile[turn_seven_entry : turn_seven_exit + 1],
            33.53,
        )

        # Propagate every corner's braking requirement backward.
        for _ in range(8):
            for index in reversed(range(self.waypoint_count)):
                next_index = (index + 1) % self.waypoint_count
                distance = np.linalg.norm(
                    path[next_index] - path[index]
                )
                allowed_speed = np.sqrt(
                    speed_profile[next_index] ** 2
                    + 2.0 * self.maximum_deceleration * distance
                )
                speed_profile[index] = min(
                    speed_profile[index],
                    allowed_speed,
                )

        return speed_profile

    def _update_progress(self, vehicle_location: np.ndarray) -> None:
        search_count = 150
        candidate_indices = (
            self.current_waypoint_idx + np.arange(search_count)
        ) % self.waypoint_count

        candidate_distances = np.linalg.norm(
            self.racing_line[candidate_indices] - vehicle_location[:2],
            axis=1,
        )
        best_candidate = int(np.argmin(candidate_distances))

        if candidate_distances[best_candidate] <= 20.0:
            self.current_waypoint_idx = int(
                candidate_indices[best_candidate]
            )
            return

        # Global recovery is used only after a respawn or large tracking error.
        all_distances = np.linalg.norm(
            self.racing_line - vehicle_location[:2],
            axis=1,
        )
        self.current_waypoint_idx = int(np.argmin(all_distances))

    def _lookahead_target(self, current_speed: float) -> np.ndarray:
        lookahead_distance = float(
            np.clip(10.0 + 0.55 * current_speed, 10.0, 50.0)
        )
        if 1380 <= self.current_waypoint_idx < 1470:
            lookahead_distance = min(lookahead_distance, 20.0)

        accumulated_distance = 0.0
        index = self.current_waypoint_idx

        for _ in range(150):
            next_index = (index + 1) % self.waypoint_count
            accumulated_distance += np.linalg.norm(
                self.racing_line[next_index] - self.racing_line[index]
            )
            index = next_index

            if accumulated_distance >= lookahead_distance:
                break

        return self.racing_line[index]

    def _pure_pursuit_steering(
        self,
        vehicle_location: np.ndarray,
        vehicle_yaw: float,
        target: np.ndarray,
    ) -> float:
        delta = target - vehicle_location[:2]

        local_y = (
            -np.sin(vehicle_yaw) * delta[0]
            + np.cos(vehicle_yaw) * delta[1]
        )
        lookahead_squared = max(float(np.dot(delta, delta)), 1e-6)

        wheelbase = 2.875
        maximum_steering_angle = 0.70

        steering_angle = np.arctan2(
            2.0 * wheelbase * local_y,
            lookahead_squared,
        )
        raw_steer = -steering_angle / maximum_steering_angle
        raw_steer = float(np.clip(raw_steer, -1.0, 1.0))

        # Smooth commands while still allowing the wheel to unwind quickly.
        smoothed_steer = 0.65 * self.previous_steer + 0.35 * raw_steer
        maximum_change = (
            0.10
            if abs(raw_steer) < abs(self.previous_steer)
            else 0.06
        )
        steer = float(
            np.clip(
                smoothed_steer,
                self.previous_steer - maximum_change,
                self.previous_steer + maximum_change,
            )
        )
        self.steer_is_unwinding = abs(steer) < abs(self.previous_steer)
        self.previous_steer = steer
        return steer

    def _longitudinal_control(
        self,
        current_speed: float,
        target_speed: float,
        steer: float,
        curvature: float,
    ):
        speed_error = target_speed - current_speed
        overspeed = -speed_error
        speed_change = current_speed - self.previous_speed
        self.previous_speed = current_speed

        turning = curvature >= 0.002 or abs(steer) >= 0.05

        current_speed_kmh = current_speed * 3.6
        maintenance_throttle = float(
            np.clip(
                0.45 + current_speed_kmh / 600.0,
                0.45,
                0.85,
            )
        )
        if turning:
            maintenance_throttle = float(
                np.clip(
                    0.60 + current_speed_kmh / 700.0,
                    0.60,
                    0.95,
                )
            )

        # Complete a short brake pulse instead of rapidly switching the brake
        # on and off every simulation tick.
        if self.brake_ticks_remaining > 0:
            if speed_error >= 0.0:
                self.brake_ticks_remaining = 0
                self.brake_release_ticks = 0
                if speed_error > 0.5:
                    return 1.0, 0.0
                return maintenance_throttle, 0.0

            self.brake_ticks_remaining -= 1
            if self.brake_ticks_remaining == 0:
                self.brake_release_ticks = 2

            if turning and overspeed < 5.0:
                return 0.25, 0.60
            return 0.0, 1.0

        # Give the previous brake command time to slow the car. Reapply
        # throttle early when speed is already falling quickly.
        if self.brake_release_ticks > 0:
            if speed_error > 0.5:
                self.brake_release_ticks = 0
                return 1.0, 0.0

            self.brake_release_ticks -= 1
            if speed_change < -0.20:
                return (0.90 if turning else 0.55), 0.0
            return 0.25, 0.0

        if speed_error > 2.0:
            return 1.0, 0.0

        if speed_error > 0.5:
            throttle = float(
                np.clip(
                    maintenance_throttle + 0.10 * speed_error,
                    0.0,
                    1.0,
                )
            )
            return throttle, 0.0

        if speed_error >= -0.5:
            return maintenance_throttle, 0.0

        # As soon as the wheel begins to unwind, favor acceleration unless
        # the car is still substantially above the planned speed.
        if turning and self.steer_is_unwinding and overspeed < 2.0:
            return 1.0, 0.0

        # Use light left-foot braking near the target so the car keeps
        # pulling through a turn rather than coasting and losing momentum.
        if overspeed < 3.0:
            throttle = 0.45 if turning else 0.15
            brake = float(np.clip(0.05 + 0.04 * overspeed, 0.0, 0.20))
            return throttle, brake

        # Larger speed errors trigger a bounded 2-5 tick braking pulse.
        # Hard braking is delayed while steering unwinds on corner exit.
        if turning and self.steer_is_unwinding and overspeed < 4.0:
            return 0.40, 0.15

        self.brake_ticks_remaining = int(
            np.clip(np.ceil(overspeed / 1.5), 2, 5)
        )
        self.brake_ticks_remaining -= 1
        if self.brake_ticks_remaining == 0:
            self.brake_release_ticks = 2
        return (0.15 if turning and overspeed < 5.0 else 0.0), 1.0

    def _corner_exit_target(
        self,
        current_index: int,
        normal_target_speed: float,
    ) -> float:
        for turn_number, (_entry, apex, exit_index) in enumerate(
            self.corner_segments,
            start=1,
        ):
            if not apex <= current_index <= exit_index:
                continue

            if turn_number == 10:
                return normal_target_speed

            exit_length = max(exit_index - apex, 1)
            progress = (current_index - apex) / exit_length
            acceleration_blend = smoothstep(progress)

            # Look beyond the corner exit. Taking the minimum keeps the car
            # from accelerating hard when another slow corner follows.
            post_corner_speed = min(
                self.speed_profile[
                    (exit_index + offset) % self.waypoint_count
                ]
                for offset in (20, 40, 60)
            )
            post_corner_speed = min(
                float(post_corner_speed),
                self.maximum_straight_speed,
            )

            apex_speed = float(self.speed_profile[apex])
            accelerating_target = (
                apex_speed
                + acceleration_blend
                * (post_corner_speed - apex_speed)
            )

            return max(
                normal_target_speed,
                accelerating_target,
            )

        return normal_target_speed

    def _log_checkpoints(
        self,
        current_speed: float,
        target_speed: float,
        throttle: float,
        brake: float,
    ) -> None:
        self.tick_count += 1
        elapsed_seconds = self.tick_count * self.seconds_per_tick

        waypoints_advanced = (
            self.current_waypoint_idx - self.previous_waypoint_idx
        ) % self.waypoint_count

        if waypoints_advanced <= 100:
            self.total_waypoints_passed += waypoints_advanced

        while self.total_waypoints_passed >= self.next_checkpoint:
            split_time = elapsed_seconds - self.previous_checkpoint_time
            print(
                f"Checkpoint {self.next_checkpoint} | "
                f"Total: {elapsed_seconds:.2f}s | "
                f"Split: {split_time:.2f}s | "
                f"Speed: {current_speed * 2.23694:.1f} mph | "
                f"Target: {target_speed * 2.23694:.1f} mph | "
                f"Throttle: {throttle:.2f} | Brake: {brake:.2f}"
            )
            self.previous_checkpoint_time = elapsed_seconds
            self.next_checkpoint += self.checkpoint_interval

        self.previous_waypoint_idx = self.current_waypoint_idx

    async def step(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        current_speed = float(np.linalg.norm(vehicle_velocity))

        self._update_progress(vehicle_location)

        target_speed = float(
            self.speed_profile[self.current_waypoint_idx]
        )
        target_speed = self._corner_exit_target(
            self.current_waypoint_idx,
            target_speed,
        )
        target_location = self._lookahead_target(current_speed)
        steer = self._pure_pursuit_steering(
            vehicle_location,
            float(vehicle_rotation[2]),
            target_location,
        )
        throttle, brake = self._longitudinal_control(
            current_speed,
            target_speed,
            steer,
            float(self.curvature_profile[self.current_waypoint_idx]),
        )

        force_full_throttle = (
            self.current_waypoint_idx < self.opening_full_throttle_end
            or (
                self.middle_full_throttle_start
                <= self.current_waypoint_idx
                < self.middle_full_throttle_end
            )
            or (
                self.curve_ten_full_throttle_start
                <= self.current_waypoint_idx
                < self.curve_ten_full_throttle_end
            )
            or (
                self.back_straight_full_throttle_start
                <= self.current_waypoint_idx
                < self.back_straight_end
            )
        )

        if force_full_throttle:
            self.brake_ticks_remaining = 0
            self.brake_release_ticks = 0
            throttle = 1.0
            brake = 0.0

        self._log_checkpoints(
            current_speed,
            target_speed,
            throttle,
            brake,
        )

        control = {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0,
        }

        await self.vehicle.apply_action(control)
        return control


