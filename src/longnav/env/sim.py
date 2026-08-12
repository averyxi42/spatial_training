import math
import numpy as np
import habitat_sim
from typing import Dict, List, Any, Optional
from pure_pursuit import PurePursuitController,ControllerConfig

class HabitatRobotSim:
    def __init__(
        self, 
        sim_settings: Dict[str, Any], 
        urdf_path: str, 
        sensor_setup: Dict[str, List[habitat_sim.sensor.SensorSpec]], 
        fixed_base: bool = False,
        max_substep_dt: float = 1.0 / 120.0  # high frequency for accurate kinematic integration
    ):
        self.urdf_path = urdf_path
        self.sensor_setup = sensor_setup
        self.fixed_base = fixed_base
        self.max_substep_dt = max_substep_dt
        self.robot = None
        self.joint_name_to_motor_id = {}
        
        # 1. Flatten sensor_setup and assign to a dummy agent configuration
        self.cfg = habitat_sim.utils.settings.make_cfg(sim_settings)
        agent_id = sim_settings.get("default_agent", 0)
        self.cfg.agents[agent_id].radius=0.28
        self.cfg.agents[agent_id].height=1.0
        all_sensor_specs = []
        for specs in self.sensor_setup.values():
            all_sensor_specs.extend(specs)
            
        self.cfg.agents[agent_id].sensor_specifications = all_sensor_specs

        # 2. Initialize simulator
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent_id = agent_id
        # self.sim.reconfigure(self.cfg)

        # 3. Setup the robot and attach sensors
        self._setup_robot_and_sensors()
        self.robot = None
    def _setup_robot_and_sensors(self):
        # Load the URDF
        aom = self.sim.get_articulated_object_manager()
        if self.robot is None:
            self.robot = aom.add_articulated_object_from_urdf(
                filepath=self.urdf_path,
                fixed_base=self.fixed_base,
                global_scale=1.0,
                mass_scale=1.0,
                force_reload=True,
                maintain_link_order=False,
                intertia_from_urdf=False,
            )
        self.robot.auto_clamp_joint_limits = True
        import magnum as mn
        self.base_rotation_offset = mn.Quaternion.rotation(
            mn.Rad(-math.pi / 2), mn.Vector3(1.0, 0.0, 0.0)
        )
        self.robot.rotation = self.base_rotation_offset
        # Map joint names to motor IDs
        self.joint_name_to_motor_id = {
            self.robot.get_link_joint_name(link_id): motor_id
            for motor_id, link_id in self.robot.existing_joint_motor_ids.items()
        }

        # Remove motors for passive joints (just like the viewer script)
        for _passive_joint in ("joint_z", "joint_pitch", "joint_roll"):
            _motor_id = self.joint_name_to_motor_id.pop(_passive_joint, None)
            if _motor_id is not None:
                self.robot.remove_joint_motor(_motor_id)
        for _lid in range(self.robot.num_links):
            if self.robot.get_link_joint_name(_lid) == "joint_body_mount":
                self.robot.set_link_friction(_lid, 0.01)
                break
        # Map link names to link IDs to easily find where to attach sensors
        link_name_to_id = {
            self.robot.get_link_name(i): i 
            for i in range(self.robot.num_links)
        }

        # Attach sensors to the corresponding robot links
        agent_node = self.sim.get_agent(self.agent_id).scene_node
        for link_name, specs in self.sensor_setup.items():
            if link_name not in link_name_to_id:
                raise ValueError(f"Link '{link_name}' not found in URDF.")
                
            link_id = link_name_to_id[link_name]
            link_scene_node = self.robot.get_link_scene_node(link_id)
            
            for spec in specs:
                # Get the instantiated sensor node from the agent's subtree
                sensor_node = agent_node.subtree_sensors.get(spec.uuid)
                if sensor_node is not None:
                    # Detach from agent and attach to the robot link
                    sensor_node.object.parent = link_scene_node
                    # Counteract the base rotation offset so the camera looks forward (-Z) and not at the floor
                    sensor_node.object.rotation = sensor_node.object.rotation = mn.Quaternion.rotation(mn.Rad(-math.pi/2), mn.Vector3(0, 1, 0))*mn.Quaternion.rotation(mn.Rad(-math.pi/2), mn.Vector3(0, 0, 1)) 
                    # Ensure the sensor is exactly at the URDF link's origin
                    sensor_node.object.translation = mn.Vector3(0, 0, 0)
    def step(
        self, 
        chassis_twist: np.ndarray, 
        joint_targets: Optional[Dict[str, float]] = None, 
        twist_frame: str = "robot",
        dt: float = 1.0 / 60.0
    ) -> dict:
        
        # Determine substepping
        num_steps = max(1, math.ceil(dt / self.max_substep_dt))
        substep_dt = dt / num_steps
        
        v_x_cmd, v_y_cmd, omega_cmd = chassis_twist
        
        for step_idx in range(num_steps):
            # 1. Update chassis velocities
            if twist_frame == "world":
                world_v_x, world_v_y = v_x_cmd, v_y_cmd
            elif twist_frame == "robot" or isinstance(twist_frame, tuple):
                # Get current heading to transform body twist to world frame
                heading = self.get_joint_state("joint_th") if twist_frame == "robot" else float(twist_frame[0])+float(twist_frame[1])*step_idx*substep_dt
                
                # Standard 2D rotation matrix
                world_v_x = v_x_cmd * math.cos(heading) - v_y_cmd * math.sin(heading)
                world_v_y = v_x_cmd * math.sin(heading) + v_y_cmd * math.cos(heading)
            else:
                raise ValueError("twist_frame must be 'world' or 'robot'")

            # Apply to motors
            for joint_name, vel in [("joint_x", world_v_x), ("joint_y", world_v_y), ("joint_th", omega_cmd)]:
                motor_id = self.joint_name_to_motor_id.get(joint_name)
                if motor_id is not None:
                    settings = self.robot.get_joint_motor_settings(motor_id)
                    settings.velocity_target = vel
                    self.robot.update_joint_motor(motor_id, settings)

            # 2. Update other joint targets (e.g., arm, head) if provided
            if joint_targets:
                for j_name, pos_target in joint_targets.items():
                    m_id = self.joint_name_to_motor_id.get(j_name)
                    if m_id is not None:
                        settings = self.robot.get_joint_motor_settings(m_id)
                        settings.position_target = pos_target
                        self.robot.update_joint_motor(m_id, settings)

            # 3. Advance physics
            self.sim.step_world(substep_dt)
            
        # Return standard observation dictionary
        return self.sim.get_sensor_observations(self.agent_id)
    
    def reset(
        self, 
        scene_id: Optional[str] = None, 
        robot_translation: Optional[np.ndarray] = None,
        robot_rotation: Optional[Any] = None,  # Accepts mn.Quaternion or [x, y, z, w] array
        snap_to_navmesh: bool = True,
        vertical_offset: Optional[float] = None, # Manual override if AABB is weird
        home_joint_positions: Optional[Dict[str, float]] = None
    ) -> dict:
        
        # 1. Handle Scene Changes and Graph Rebuild
        if scene_id is not None and scene_id != self.cfg.sim_cfg.scene_id:
            self.cfg.sim_cfg.scene_id = scene_id
            self.sim.reconfigure(self.cfg)
        else:
            self.sim.reset()
            self._setup_robot_and_sensors()

        # 2. Process Rotation (Convert [x, y, z, w] to mn.Quaternion)
        if robot_rotation is not None:
            if isinstance(robot_rotation, (np.ndarray, list, tuple)):
                if len(robot_rotation) != 4:
                    raise ValueError("robot_rotation array must have 4 elements: [x, y, z, w]")
                import magnum as mn
                x, y, z, w = robot_rotation
                robot_rotation = mn.Quaternion(mn.Vector3(x, y, z), w)
            self.robot.rotation = robot_rotation* self.base_rotation_offset
        else:
            self.robot.rotation = self.base_rotation_offset

        # 3. Handle NavMesh Snapping & Vertical Offsets
        if robot_translation is not None:
            if snap_to_navmesh and self.sim.pathfinder.is_loaded:
                safe_point = self.sim.pathfinder.snap_point(robot_translation)
                if not np.isnan(safe_point[0]):
                    robot_translation = safe_point
            
            # Compute dynamic vertical offset if not manually provided
            if vertical_offset is None:
                aabb = self.robot.root_scene_node.cumulative_bb
                # If lowest point (min.y) is negative, we need to shift the robot UP
                vertical_offset = -aabb.min.y if aabb.min.y < 0 else 0.0

            # Apply translation with offset (Habitat is Y-up)
            robot_translation[1] += vertical_offset
            self.robot.translation = robot_translation
            
        # 4. Clear physics momentum & set home joints
        # Zero out velocities and forces to prevent flying away after teleport
        num_dofs = len(self.robot.joint_velocities)
        self.robot.joint_velocities = [0.0] * num_dofs
        self.robot.joint_forces = [0.0] * num_dofs

        if home_joint_positions:
            for j_name, pos_target in home_joint_positions.items():
                m_id = self.joint_name_to_motor_id.get(j_name)
                if m_id is not None:
                    # Instantly update the kinematic state
                    l_id = self.robot.existing_joint_motor_ids[m_id]
                    pos_offset = self.robot.get_link_joint_pos_offset(l_id)
                    # joint_name = self.robot.get_link_joint_name(l_id)
                    self.robot.joint_positions[pos_offset] = pos_target
                    
                    # Update the motor target so it doesn't try to snap back
                    settings = self.robot.get_joint_motor_settings(m_id)
                    settings.position_target = pos_target
                    self.robot.update_joint_motor(m_id, settings)
        # self.robot.awake = True
        return self.sim.get_sensor_observations(self.agent_id)
    #TODO: fix this
    def get_joint_state(self, joint_name: str, is_vel: bool = False) -> float:
        """
        Retrieves the position or velocity of a specific joint by its name.
        """
        motor_id = self.joint_name_to_motor_id.get(joint_name)
        if motor_id is None:
            raise ValueError(f"Joint '{joint_name}' not found in motor mappings.")
            
        link_id = self.robot.existing_joint_motor_ids[motor_id]
        print(self.robot.joint_velocities)
        if is_vel:
            # joint_velocities are indexed by DOF id
            dof_offset = self.robot.get_link_dof_offset(link_id)
            return self.robot.joint_velocities[dof_offset]
        else:
            # joint_positions are indexed by position offset
            pos_offset = self.robot.get_link_joint_pos_offset(link_id)
            return self.robot.joint_positions[pos_offset]
        
    def get_2d_pose_and_twist(self, virtual_heading = None) -> tuple[np.ndarray, np.ndarray]:
        import math
        import numpy as np
        import magnum as mn

        jx = self.get_joint_state("joint_x")
        jy = self.get_joint_state("joint_y")
        jth = self.get_joint_state("joint_th") if virtual_heading is None else virtual_heading  # Adjust for URDF orientation
        
        # Retrieve current velocity
        vx = self.get_joint_state("joint_x", is_vel=True)#*1024
        vy = self.get_joint_state("joint_y", is_vel=True)#*1024
        vth = self.get_joint_state("joint_th", is_vel=True)
        
        rz,rx = self.robot.translation[2],self.robot.translation[0]
        v_forward = vx * math.cos(jth) + vy * math.sin(jth)
        
        # limit jth to [-pi, pi]
        jth = (jth + math.pi) % (2 * math.pi) - math.pi
        return np.array([jx+rx,jy-rz, jth]), np.array([v_forward, vth])
    def navmesh_config_and_recompute(self) -> None:
        """
        This method is setup to be overridden in for setting config accessibility
        in inherited classes.
        """
        self.navmesh_settings = habitat_sim.NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_height = self.cfg.agents[self.agent_id].height
        self.navmesh_settings.agent_radius = self.cfg.agents[self.agent_id].radius
        self.navmesh_settings.include_static_objects = True
        self.navmesh_settings.agent_max_climb = 0.2
        # print(self.navmesh_settings)
        self.sim.recompute_navmesh(
            self.sim.pathfinder,
            self.navmesh_settings,
        )
    def draw_axes(self, translation, axis_len=1.0):
        lr = self.sim.get_debug_line_render()
        lr.set_line_width(5)

        import magnum as mn
        opacity = 1.0   
        red = mn.Color4(1.0, 0.0, 0.0, opacity)
        green = mn.Color4(0.0, 1.0, 0.0, opacity)
        blue = mn.Color4(0.0, 0.0, 1.0, opacity)
        white = mn.Color4(1.0, 1.0, 1.0, opacity)
        # draw axes with x+ = red, y+ = green, z+ = blue
        lr.draw_transformed_line(translation, mn.Vector3(axis_len, 0, 0)+translation, red)
        lr.draw_transformed_line(translation, mn.Vector3(0, axis_len, 0)+translation, green)
        lr.draw_transformed_line(translation, mn.Vector3(0, 0, axis_len)+translation, blue)
    def draw_path(self,path_3d):
        lr = self.sim.get_debug_line_render()
        lr.set_line_width(5)
        import magnum as mn
        opacity = 1.0   
        offset = mn.Vector3(0, 0.2, 0)
        white = mn.Color4(1.0, 0.0, 1.0, opacity)
        for i in range(len(path_3d)-1):
            lr.draw_transformed_line(path_3d[i]+offset, path_3d[i+1]+offset, white)
from habitat_sim.utils.settings import default_sim_settings
def main():
    import imageio
    # 1. Deduce settings from your viewer script
    sim_settings = default_sim_settings.copy()
    sim_settings["scene"] = "/home/avery/codes/habitat/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"#"/home/avery/codes/habitat/data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
    # sim_settings["scene"] = "/home/avery/codes/habitat/data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
    sim_settings["enable_physics"] = True
    sim_settings["window_width"] = 640
    sim_settings["window_height"] = 480
    sim_settings["default_agent"] = 0

    urdf_path = "/home/avery/codes/habitat/tidybot_ros/src/tidybot_description/urdf/tidybot_base_minimal.urdf"

    # 2. Configure the sensor
    color_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.sensor.SensorType.COLOR
    color_sensor_spec.resolution = [sim_settings["window_height"], sim_settings["window_width"]]
    
    # NOTE: You may need to change "base_link" to the specific camera link name in your URDF
    sensor_setup = {"base_camera_link": [color_sensor_spec]}

    # 3. Initialize Simulator
    print("Initializing simulator...")
    robot_sim = HabitatRobotSim(
        sim_settings=sim_settings,
        urdf_path=urdf_path,
        sensor_setup=sensor_setup,
        fixed_base=True
    )
    robot_sim.navmesh_config_and_recompute()
    robot_sim.sim.navmesh_visualization = True
    import random

    # Seed the simulator's internal RNG with a random integer
    robot_sim.sim.seed(random.randint(1, 999999))
    # Reset and snap to NavMesh
    print("Resetting and snapping to NavMesh...")
    pathfinder = robot_sim.sim.pathfinder

    # 2. Sample a random point if the NavMesh is loaded, otherwise fallback
    if pathfinder.is_loaded:
        initial_position = pathfinder.get_random_navigable_point()
        print(f"Sampled random initial position: {initial_position}")
    else:
        print("Warning: NavMesh not loaded. Falling back to origin.")
        initial_position = np.array([0.0, 0.0, 0.0])
    # initial_position = np.array([1.30228, 0.209919, 15.315])
    
    #goal: [-19.06322098  -2.51216865]
    # 3. Pass it to the reset function
    initial_obs = robot_sim.reset(
        scene_id=sim_settings["scene"],
        robot_translation=initial_position, 
        snap_to_navmesh=True,
        vertical_offset=None,
    )
    # 4. Execute a driving pattern
    print("Driving robot and recording observations...")
    frames = []
    fps = 30
    duration_seconds = 10
    

    import time
    import math
    
    # 1. Sample a random reachable goal and find the shortest path
    path = habitat_sim.ShortestPath()
    
    # We assume the robot was just reset, so its root translation is its exact starting point
    path.requested_start = robot_sim.robot.translation 
    
    path.requested_end = robot_sim.sim.pathfinder.get_random_navigable_point()
    print(f"height difference: {path.requested_start[1]-path.requested_end[1]}")
    # disallow height difference greater than 0.5 meters
    max_height_diff = 0.2
    found = robot_sim.sim.pathfinder.find_path(path) and abs(path.requested_start[1]-path.requested_end[1])<max_height_diff
    for i in range(10):
        if found:
            break
        print("Resampling goal point...")
        path.requested_end = robot_sim.sim.pathfinder.get_random_navigable_point()
        found = robot_sim.sim.pathfinder.find_path(path) and abs(path.requested_start[1]-path.requested_end[1])<max_height_diff
        
    if not found:
        print("Path not found! Try another start/end point.")
    else:

        height_diff = path.requested_start[1]-path.requested_end[1]
        print(f"Path found with {len(path.points)} points. Following...")
        
        # 2. Extract the 2D path (Habitat X and Z)
        path_3d = np.array(path.points)
        path_2d = path_3d[:, [0, 2]]*np.array([[1.0,-1.0]]) 
        
        # 3. Configure the Controller
        config = ControllerConfig(
            min_look_ahead_distance=0.05,
            max_look_ahead_distance=1.0,
            look_ahead_time=1.0,
            v_max=2.0,
            v_min=0.0,
            w_max=10.0,
            w_min=-10.0,
            a_max=3.0,
            aw_max=100.0,
            dt=1.0/fps,
            approach_velocity_scaling_dist=0.5,
            min_approach_linear_velocity=0.1,
            goal_tolerance_dist=0.15,  # Matches the dataclass spelling
            regulated_linear_scaling_min_radius=0.01,
            regulated_linear_scaling_min_speed=0.1
        )
        controller = PurePursuitController(config, method_name="dwpp")
        controller.set_path(path_2d)
        goal_pos = path_2d[-1]
        print(f"Initial Position: {path_2d[0]}, Goal position: {goal_pos}")
        print(f"goal direction: {np.arctan2(goal_pos[1]-path_2d[0][1],goal_pos[0]-path_2d[0][0])}")
        t0 = time.perf_counter()
        initial_dist_to_goal = np.linalg.norm(path_2d[0] - goal_pos)
        min_dist_to_goal = initial_dist_to_goal
        # 4. Drive the robot along the path
        # first rotate in place to face the next waypoint
        current_pose, controller_velocity = robot_sim.get_2d_pose_and_twist(virtual_heading=None)
        target = [path_2d[1][0], path_2d[1][1]]
        goal_direction = np.arctan2(target[1]-current_pose[1],target[0]-current_pose[0])

        virtual_heading = goal_direction
        virtual_angular_velocity = 0.0
        
        for _ in range(duration_seconds * fps):
            # 1. Get ground truth pose and velocity directly
            current_pose, controller_velocity = robot_sim.get_2d_pose_and_twist(virtual_heading=virtual_heading)
            virtual_pose = np.array([current_pose[0], current_pose[1], virtual_heading])
            
            virtual_velocity = np.array([controller_velocity[0], virtual_angular_velocity])
            # We only need [v_forward, omega] for the controller
            
            # 2. Check for completion
            dist_to_goal = np.linalg.norm(current_pose[:2] - goal_pos)
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
            goal_reached = dist_to_goal < config.goal_tolerance_dist
            if goal_reached:
                print("Goal reached successfully!")
            else:
                vw = virtual_heading - robot_sim.get_joint_state("joint_th")
                #clamp vw to something sane
                vw = max(-1.5, min(1.5, vw))
            # print(f"current_pose: {current_pose}, controller_velocity: {controller_velocity}, goal_direction: {np.arctan2(goal_pos[1]-current_pose[1],goal_pos[0]-current_pose[0])}, dist_to_goal: {dist_to_goal}")
            # 3. Compute optimal velocities
            next_velocity_ref, debug_info = controller.compute_velocity(virtual_pose, virtual_velocity)
            v_ref, w_ref = next_velocity_ref
            # 4. Apply commands in the robot frame
            twist_command = np.array([v_ref, 0.0, vw]) if not goal_reached else np.array([0.0, 0.0, 0.0])  # Only forward velocity and yaw difference
            robot_sim.draw_axes(path.requested_start, axis_len=1)
            robot_sim.draw_axes(path.requested_end, axis_len=1)
            robot_sim.draw_path(path_3d)
            obs = robot_sim.step(
                chassis_twist=twist_command,
                twist_frame=(virtual_heading, virtual_angular_velocity),
                dt=1.0 / fps
            )
            virtual_heading += virtual_angular_velocity * (1.0 / fps)
            virtual_angular_velocity = w_ref
            rgb_frame = obs["color_sensor"][..., :3]
            frames.append(rgb_frame)
            
    t1 = time.perf_counter()
    print(f"Finished driving for {duration_seconds} seconds. Total frames captured: {len(frames)}")
    print(f"Simulation time: {t1 - t0:.2f} seconds. Speedup: {duration_seconds / (t1 - t0):.2f}x real-time.")
    # 5. Save to file
    output_file = "robot_drive_output.mp4"
    print(f"Saving video to {output_file}...")
    imageio.mimsave(output_file, frames, fps=fps)
    print("Done!")
    print(f"Initial distance to goal: {initial_dist_to_goal}")
    print(f"Minimum distance to goal: {min_dist_to_goal}")

if __name__ == "__main__":
    main()