import base64
import os
import zlib
from typing import List, Tuple

import numpy as np

import roar_py_interface


GRIP = 2.20
GRIP_FRACTION = 0.95

# The offsets are scaled toward the centerline before use. The generated line is
# anchored within 1.5 m of a proven one and clamped at 3.5 m, so it is not the
# line that put the car off; pulling it in was about leaving room for a tracking
# error the offline model could not predict.
#


RACING_LINE_SCALE = 0.70
MAXIMUM_SPEED = 89.4
MINIMUM_CORNER_SPEED = 17.0
# Curvature below this is treated as straight, so real straights keep the full
# target instead of being slowed by survey noise.
CURVATURE_DEADBAND = 0.002

# Braking, m/s^2. Rises with speed because aerodynamic drag helps at the top
# end: 13.1 / 14.3 / 15.4 are the reference entry's 170 / 185 / 200 in
# (km/h)^2/m converted to SI, and they bracket our own measured p95 of 9.8.
DECELERATION = ((0.0, 13.1), (58.3, 14.3), (63.9, 15.4))
# Ticks of travel before braking bites, subtracted from the distance available.
# Braking this much earlier is cheap insurance and costs little: at 80 m/s it
# gives up 24 m of the roughly 250 m needed to reach a slow corner.
REACTION_TICKS = 6.0
# Far enough to see a 200 mph to 52 mph stop, which needs about 290 m, plus the
# reaction allowance.
BRAKING_HORIZON = 360.0

LOOKAHEAD_PER_KMH = 0.25
LOOKAHEAD_MIN = 15.0
LOOKAHEAD_MAX = 70.0

# The reference's pursuit constants: an effective wheelbase well over the car's
# real 2.875 m, and an output gain on top.
#
# Neither is a mistake to be corrected. On a circular path the aim point sits at
# sin(alpha) = lookahead / (2 * radius), so the whole law collapses to
# gain * atan(effective_wheelbase / radius) and the lookahead cancels out. Times
# the car's steering range that is 1.72x the front angle the bicycle model says
# the corner needs.
#
# A real car does need more than the bicycle model, because the front tyres have
# to run at a slip angle to make force, and at the 2+ g these corners are taken
# at that allowance is a large part of the angle. The old law here commanded the
# bicycle angle and nothing else, so it asked for 58% of what the corner took and
# the feedback terms had to find the rest, which is why they were wound up far
# enough to oscillate. Getting the feedforward right is most of why this swap
# should hold.
#
# This is also why drive_offline.py integrates yaw at a 4.935 m cornering
# wheelbase rather than the real 2.875. At the true one the model needs no
# understeer allowance, so it flattered the old law and would penalise this one.
PURSUIT_WHEELBASE = 4.7
PURSUIT_GAIN = 1.5

# Below this the car is still planted at the spawn. Ticks 1-31 of an earlier run
# sat at 0 mph building 0.017 of steer from a 2 degree spawn misalignment, then
# launched already yawing.
LAUNCH_STEER_SPEED = 4.0

# --- Per-corner tuning ------------------------------------------------------
#
# Corner (entry, apex, exit) triples in official-waypoint indices.
CORNER_SEGMENTS = [
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

# Overrides keyed by turn number, empty until a clean lap says which corner
# needs what. The reference entry carries a dozen of these tied to indices in
# its own waypoint set; ours are keyed by turn so they mean something. Raising
# a turn's grip lets it be taken faster; raising its steer scale makes the car
# turn harder there.
CORNER_GRIP: dict = {}
CORNER_STEER_SCALE: dict = {}

# Windows, as (waypoints behind, waypoints ahead), the preview curvature is
# measured over. Waypoints are about 2 m apart, so these span 60 to 89 m of
# track, and the tightest reading across all of them wins.
#
# That ordering is what makes wide windows safe. A three-point fit across an
# S-turn samples nearly collinear points and reports a straight: at turn 5 a
# +/-48 waypoint fit reads 1587 m of radius where a +/-4 fit reads 45 m. Taking
# a median of wide spans let that cancellation raise the target to 118 mph for a
# bend needing 6.3 g and put the car into the wall. Taking the tightest reading
# means a cancelled window can only ever be discarded by a more conservative one.
CURVATURE_WINDOWS = (
    (15, 15),
    (22, 22),
    (30, 30),
    (7, 22),
    (22, 7),
    (11, 33),
    (33, 11),
)
# Narrow enough to measure the arc the car is on right now, wide enough to
# ignore the 2 m survey jitter that makes a +/-6 m estimate useless.
LOCAL_WINDOWS = ((8, 8),)

# BEGIN GENERATED RACING LINE -- rewritten by tools/generate_racing_line.py
RACING_LINE_OFFSETS = (
    "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"
)
# END GENERATED RACING LINE


def normalize_rad(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def decode_offsets(blob: str) -> np.ndarray:
    """Undo the generator's encoding: base64, zlib, int16 millimetre deltas.

    Deltas because the offsets vary smoothly, which makes them compress to
    about a quarter of the raw floats. Millimetres because the curvature is
    measured over 16 m or more and cannot see half a millimetre.
    """
    try:
        raw = zlib.decompress(base64.b64decode(blob))
    except Exception as error:
        # The likely cause is this file having been copied incompletely, which
        # is worth naming: the alternative is a stack trace about checksums.
        raise ValueError(
            "the embedded racing line will not decode (%s) -- this file was "
            "probably copied incompletely, copy it again whole" % error
        )
    deltas = np.frombuffer(raw, dtype=np.int16)
    return np.cumsum(deltas.astype(np.int64)) / 1000.0


def left_normals(path: np.ndarray) -> np.ndarray:
    """Unit normals from central-difference tangents."""
    tangents = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    tangents /= np.maximum(
        np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9
    )
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def curvature_profile(path: np.ndarray, windows) -> np.ndarray:
    """Menger curvature over several spans, tightest reading wins."""
    count = len(path)
    readings = []

    for behind, ahead in windows:
        a = np.roll(path, behind, axis=0)
        b = path
        c = np.roll(path, -ahead, axis=0)

        cross = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        denominator = (
            np.linalg.norm(b - a, axis=1)
            * np.linalg.norm(c - b, axis=1)
            * np.linalg.norm(c - a, axis=1)
        )
        readings.append(
            np.divide(
                2.0 * cross,
                denominator,
                out=np.zeros(count, dtype=float),
                where=denominator > 1e-6,
            )
        )

    return np.max(np.array(readings), axis=0)


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
        self.maneuverable_waypoints = maneuverable_waypoints
        self.official_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor

        self.reported_failure = False
        self.respawn_count = 0

    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()

        # Telemetry is a diagnostic and nothing reads it while driving, so a
        # directory that cannot be written to on the competition machine must
        # not be able to end the run before it starts.
        try:
            self.telemetry_file = open(
                os.path.join(os.path.dirname(__file__), "telemetry_log.csv"),
                "w",
            )
            self.telemetry_file.write(
                "tick,seconds,waypoint,speed_mph,target_mph,throttle,brake,"
                "steer,curvature,cross_track_m,heading_error_deg,preview_m,"
                "offset_m,x,y,turn,radius_m,lat_g\n"
            )
        except OSError as error:
            self.telemetry_file = None
            # print("telemetry disabled: %s" % error)

        self.centerline = np.array(
            [waypoint.location[:2] for waypoint in self.official_waypoints],
            dtype=float,
        )
        # waypoint.lane_width is deliberately not read. It reports a flat
        # placeholder here, and trusting it as track width is what put the car
        # into the wall at turn 5: the corridor is set offline against a proven
        # line instead.
        self.waypoint_count = len(self.centerline)

        self._load_racing_line()

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
        self.preview_distance = 0.0
        # print("initialize() complete -- logging to telemetry_log.csv")

    def _load_racing_line(self) -> None:
        """Rebuild the line and its corner speed limits from the offsets.

        Only the lateral offsets are carried, because everything else follows
        from them: the line is the centreline pushed sideways, and its
        curvature is a function of the line. That keeps this a single file with
        nothing to copy alongside it.
        """
        offsets = decode_offsets(RACING_LINE_OFFSETS) * RACING_LINE_SCALE

        # The offsets are indexed by official waypoint, so a mismatch would
        # have the car steering toward a point meant for a different track.
        # Say so here rather than driving into a wall at turn 1.
        if len(offsets) != self.waypoint_count:
            raise ValueError(
                "the embedded line has %d waypoints, the race supplied %d -- "
                "regenerate with tools/generate_racing_line.py"
                % (len(offsets), self.waypoint_count)
            )

        self.lateral_offsets = offsets
        self.racing_line = (
            self.centerline + left_normals(self.centerline) * offsets[:, None]
        )

        tangents = np.roll(self.racing_line, -1, axis=0) - np.roll(
            self.racing_line, 1, axis=0
        )
        tangents /= np.maximum(
            np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9
        )
        self.racing_line_headings = np.arctan2(tangents[:, 1], tangents[:, 0])
        self.racing_line_normals = np.column_stack(
            (-tangents[:, 1], tangents[:, 0])
        )

        # A preview reading, tightest bend within about 66 m. It reads high
        # while the car is still straight, which is what makes it safe to plan
        # braking against.
        self.curvature_profile = curvature_profile(
            self.racing_line, CURVATURE_WINDOWS
        )
        # The bend the car is in right now, over +/-16 m.
        self.line_curvature = curvature_profile(self.racing_line, LOCAL_WINDOWS)
        # Which turn each station belongs to, 0 between corners. This is the
        # tuning key: the reference entry hangs its overrides off raw indices
        # in its own waypoint set, which means nothing to a reader.
        self.corner_segments = list(CORNER_SEGMENTS)
        self.turn_at = np.zeros(self.waypoint_count, dtype=int)
        for number, (entry, _, exit_) in enumerate(self.corner_segments, 1):
            self.turn_at[entry : exit_ + 1] = number

        # Distance from each station to the next, and the cumulative station,
        # so the braking scan can work in metres without a loop.
        self.step_length = np.linalg.norm(
            np.roll(self.racing_line, -1, axis=0) - self.racing_line, axis=1
        )
        self.station = np.concatenate(
            ([0.0], np.cumsum(self.step_length)[:-1])
        )
        self.lap_length = float(self.step_length.sum())

        grip = GRIP_FRACTION * np.array(
            [
                CORNER_GRIP.get(int(turn), GRIP)
                for turn in self.turn_at
            ],
            dtype=float,
        )
        # The speed the line allows at each station, ignoring braking. The live
        # controller works out when to start slowing for these.
        bend = np.maximum(self.curvature_profile, self.line_curvature)
        limit = np.full(self.waypoint_count, MAXIMUM_SPEED, dtype=float)
        curved = bend >= CURVATURE_DEADBAND
        limit[curved] = np.sqrt(grip[curved] * 9.81 / bend[curved])
        self.corner_speed_limit = np.clip(
            limit, MINIMUM_CORNER_SPEED, MAXIMUM_SPEED
        )

        # print(
        #     "racing line: %d waypoints, peak offset %.2f m, "
        #     "corner limits %.1f to %.1f mph at %.2f g"
        #     % (
        #         self.waypoint_count,
        #         np.abs(offsets).max(),
        #         self.corner_speed_limit.min() / 0.44704,
        #         self.corner_speed_limit.max() / 0.44704,
        #         GRIP * GRIP_FRACTION,
        #     )
        # )

    def _update_progress(self, vehicle_location: np.ndarray) -> None:
        """Advance the station index, searching near the last one.

        A window rather than a global search, so the index cannot jump
        backwards onto a part of the line that happens to pass close by. It
        widens to the whole lap when nothing nearby fits, which is what a
        respawn looks like.
        """
        window = (
            self.current_waypoint_idx + np.arange(-10, 60)
        ) % self.waypoint_count
        distances = np.linalg.norm(
            self.racing_line[window] - vehicle_location[:2], axis=1
        )
        best = int(np.argmin(distances))

        if distances[best] > 15.0:
            self.current_waypoint_idx = int(
                np.argmin(
                    np.linalg.norm(
                        self.racing_line - vehicle_location[:2], axis=1
                    )
                )
            )
        else:
            self.current_waypoint_idx = int(window[best])

    def _path_error(
        self,
        vehicle_location: np.ndarray,
        vehicle_yaw: float,
    ) -> Tuple[float, float]:
        index = self.current_waypoint_idx
        delta = vehicle_location[:2] - self.racing_line[index]
        cross_track = float(np.dot(delta, self.racing_line_normals[index]))
        heading_error = normalize_rad(
            float(self.racing_line_headings[index]) - vehicle_yaw
        )
        return cross_track, heading_error

    def _aim_point(self, distance: float) -> np.ndarray:
        """The point on the racing line `distance` metres along from here.

        Walks the station spacing rather than indexing a fixed number of
        waypoints ahead, so the aim distance is the metres it claims to be. The
        reference reaches its distance by counting waypoints, which only works
        while their spacing is uniform.
        """
        index = self.current_waypoint_idx
        travelled = 0.0
        for _ in range(self.waypoint_count):
            step = float(self.step_length[index])
            if travelled + step >= distance:
                # Interpolate within the segment, so the aim point moves
                # smoothly instead of jumping station to station.
                fraction = (distance - travelled) / max(step, 1e-9)
                here = self.racing_line[index]
                nxt = self.racing_line[(index + 1) % self.waypoint_count]
                return here + (nxt - here) * fraction
            travelled += step
            index = (index + 1) % self.waypoint_count
        return self.racing_line[index]

    def _steering(
        self,
        current_speed: float,
        vehicle_location: np.ndarray,
        vehicle_yaw: float,
    ) -> float:
        """Pure pursuit onto the racing line.

        One term, no gains to balance against each other. The aim point is on
        the line, so being off the line and being mis-aimed both show up in the
        same angle, and the lookahead alone decides how hard the car is asked to
        converge.
        """
        lookahead = float(
            np.clip(
                LOOKAHEAD_PER_KMH * current_speed * 3.6,
                LOOKAHEAD_MIN,
                LOOKAHEAD_MAX,
            )
        )
        self.preview_distance = lookahead

        if current_speed < LAUNCH_STEER_SPEED:
            return 0.0

        target = self._aim_point(lookahead)
        offset = target - vehicle_location[:2]
        span = float(np.linalg.norm(offset))
        if span < 1e-6:
            return 0.0

        # Angle between where the car points and where the aim point is. Only
        # its sine is used, so wrapping it is unnecessary.
        alpha = vehicle_yaw - float(np.arctan2(offset[1], offset[0]))

        # Negative steer turns left. An aim point to the left puts its bearing
        # above the yaw, making alpha and therefore this negative already.
        angle = PURSUIT_GAIN * float(
            np.arctan2(2.0 * PURSUIT_WHEELBASE * np.sin(alpha), span)
        )
        turn = int(self.turn_at[self.current_waypoint_idx])
        angle *= CORNER_STEER_SCALE.get(turn, 1.0)

        return float(np.clip(angle, -1.0, 1.0))

    def _recommended_speed(self, current_speed: float) -> float:
        """The fastest we can be going now and still make every corner ahead.

        Asked live rather than baked into a profile, because the answer depends
        on the speed the car actually has: the reaction allowance below is a
        distance, and at 80 m/s it is three times what it is at 27.
        """
        reaction = REACTION_TICKS * self.seconds_per_tick * current_speed
        decel_speeds, decel_values = zip(*DECELERATION)
        deceleration = float(
            np.interp(current_speed, decel_speeds, decel_values)
        )

        index = self.current_waypoint_idx
        count = int(BRAKING_HORIZON / 2.0) + 1
        ahead = (index + np.arange(1, count + 1)) % self.waypoint_count
        distance = (
            self.station[ahead] - self.station[index]
        ) % self.lap_length

        limits = self.corner_speed_limit[ahead]
        braking_distance = np.maximum(distance - reaction, 0.0)
        allowed = np.sqrt(
            limits**2 + 2.0 * deceleration * braking_distance
        )

        return float(
            min(
                self.corner_speed_limit[index],
                allowed.min(),
                MAXIMUM_SPEED,
            )
        )

    def _throttle_brake(
        self,
        current_speed: float,
        target_speed: float,
    ) -> Tuple[float, float]:
        """Proportional on the speed error, with a coasting band between.

        The band matters more than the gains: without it the car alternates
        throttle and brake every tick at the corner limit, which upsets the
        platform far more than being 0.5 m/s off target costs.
        """
        error = target_speed - current_speed

        if error > 0.5:
            return float(np.clip(error / 2.0, 0.0, 1.0)), 0.0
        if error > -0.5:
            return 0.15, 0.0
        return 0.0, float(np.clip((-error - 0.5) / 2.0, 0.0, 1.0))

    def _log_telemetry(
        self,
        elapsed_seconds: float,
        vehicle_location: np.ndarray,
        current_speed: float,
        target_speed: float,
        throttle: float,
        brake: float,
        steer: float,
        cross_track_error: float,
        heading_error: float,
    ) -> None:
        if self.telemetry_file is None:
            return

        index = self.current_waypoint_idx
        bend = float(self.line_curvature[index])
        radius = 1.0 / bend if bend > 1e-6 else 99999.0
        self.telemetry_file.write(
            f"{self.tick_count},"
            f"{elapsed_seconds:.2f},"
            f"{index},"
            f"{current_speed * 2.23694:.1f},"
            f"{target_speed * 2.23694:.1f},"
            f"{throttle:.2f},"
            f"{brake:.2f},"
            f"{steer:.3f},"
            f"{self.curvature_profile[index]:.5f},"
            f"{cross_track_error:.2f},"
            f"{np.degrees(heading_error):.2f},"
            f"{self.preview_distance:.1f},"
            f"{self.lateral_offsets[index]:.2f},"
            f"{vehicle_location[0]:.2f},"
            f"{vehicle_location[1]:.2f},"
            f"{int(self.turn_at[index])},"
            f"{radius:.1f},"
            f"{current_speed ** 2 * bend / 9.81:.2f}\n"
        )
        if self.tick_count % 20 == 0:
            self.telemetry_file.flush()

    def _log_checkpoints(
        self,
        current_speed: float,
        target_speed: float,
        throttle: float,
        brake: float,
    ) -> float:
        self.tick_count += 1
        elapsed_seconds = self.tick_count * self.seconds_per_tick

        waypoints_advanced = (
            self.current_waypoint_idx - self.previous_waypoint_idx
        ) % self.waypoint_count

        if waypoints_advanced <= 100:
            self.total_waypoints_passed += waypoints_advanced
        else:
            # The car did not drive here, it was put here: a respawn after a
            # collision. Say so, or the run goes quiet at the crash and the
            # checkpoint splits after it silently describe a different lap.
            # print(
            #     f">>> RESPAWN at {elapsed_seconds:.2f}s: waypoint "
            #     f"{self.previous_waypoint_idx} -> {self.current_waypoint_idx}"
            #     f" (speed {current_speed * 2.23694:.1f} mph)"
            # )
            self.respawn_count += 1
            self.previous_checkpoint_time = elapsed_seconds

        while self.total_waypoints_passed >= self.next_checkpoint:
            split_time = elapsed_seconds - self.previous_checkpoint_time
            # print(
            #     f"Checkpoint {self.next_checkpoint} | "
            #     f"Total: {elapsed_seconds:.2f}s | "
            #     f"Split: {split_time:.2f}s | "
            #     f"Speed: {current_speed * 2.23694:.1f} mph | "
            #     f"Target: {target_speed * 2.23694:.1f} mph | "
            #     f"Throttle: {throttle:.2f} | Brake: {brake:.2f}"
            # )
            self.previous_checkpoint_time = elapsed_seconds
            self.next_checkpoint += self.checkpoint_interval

        self.previous_waypoint_idx = self.current_waypoint_idx
        return elapsed_seconds

    async def step(self) -> None:
        """Drive one tick, making any failure loud rather than silent.

        A bare exception here stops the console dead and leaves a partial
        telemetry file with no explanation, which is indistinguishable from the
        car simply having stopped. Report it once, keep the data, then re-raise
        so behaviour is unchanged.
        """
        try:
            return await self._drive()
        except Exception:
            if not getattr(self, "reported_failure", False):
                self.reported_failure = True
                import traceback

                # print("=" * 70)
                # print(
                #     "step() raised at tick %s, waypoint %s"
                #     % (
                #         getattr(self, "tick_count", "?"),
                #         getattr(self, "current_waypoint_idx", "?"),
                #     )
                # )
                # traceback.print_exc()
                # print("=" * 70)
            try:
                self.telemetry_file.flush()
            except Exception:
                pass
            raise

    async def _drive(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        current_speed = float(np.linalg.norm(vehicle_velocity))

        self._update_progress(vehicle_location)

        # Measured for the telemetry only. Pure pursuit reads the geometry to its
        # aim point directly, so neither of these feeds the steering; they are
        # here because a lap that goes wrong is diagnosed from them.
        cross_track_error, heading_error = self._path_error(
            vehicle_location,
            float(vehicle_rotation[2]),
        )
        steer = self._steering(
            current_speed,
            vehicle_location,
            float(vehicle_rotation[2]),
        )
        target_speed = self._recommended_speed(current_speed)
        throttle, brake = self._throttle_brake(current_speed, target_speed)

        elapsed_seconds = self._log_checkpoints(
            current_speed,
            target_speed,
            throttle,
            brake,
        )
        self._log_telemetry(
            elapsed_seconds,
            vehicle_location,
            current_speed,
            target_speed,
            throttle,
            brake,
            steer,
            cross_track_error,
            heading_error,
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