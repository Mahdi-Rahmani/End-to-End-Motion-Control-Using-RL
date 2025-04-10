#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# Copyright (c) 2025: Updated for CARLA 0.9.14
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

def controller_start_random_location(controller, world):
    """Start walker controller with a random destination"""
    controller.start()
    loc = world.get_random_location_from_navigation()
    if loc is not None:
        controller.go_to_location(loc)
    controller.set_max_speed(1 + random.random())  # Speed between 1 and 2

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    # Add waypoint distance threshold for termination
    self.waypoint_distance_threshold = params.get('waypoint_distance_threshold', 7)
    # Add parameters for reward calculation
    self.target_speed_kmh = params.get('target_speed_kmh', 30.0)
    self.waypoint_target_distance = params.get('waypoint_target_distance', 10.0)
    self.look_ahead_waypoint_index = params.get('look_ahead_waypoint_index', 2)

    # Add jaywalking configuration
    self.jaywalking_pedestrians = params.get('jaywalking_pedestrians', True)
    self._jaywalkers = []
    self._jaywalker_update_counters = {}

    # Add new parameters for termination configuration
    self.terminate_on_lane_departure = params.get('terminate_on_lane_departure', False)
    self.lane_departure_factor = params.get('lane_departure_factor', 1.0)

    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    
    # Added parameters for CARLA 0.9.14
    self.metadata = {
      'render_modes': ['human'],
      'render.modes': ['human']
    }
    self.host = params.get('host', 'localhost')
    self.port = params.get('port', 3000)
    self.tm_port = params.get('tm_port', 8000)
    self.sync_mode = params.get('sync', True)
    self.rendering = params.get('rendering', True)
    
    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[79, -56, 0]]
    else:
      self.dests = None

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    observation_space_dict = {
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      # Extend the ranges to ensure they can hold all possible values
      'state': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    }
    if self.pixor:
      observation_space_dict.update({
        'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
        })
    self.observation_space = spaces.Dict(observation_space_dict)

    # Connect to carla server and get world object
    print('connecting to Carla server at {}:{}...'.format(self.host, self.port))
    self.client = carla.Client(self.host, self.port)
    self.client.set_timeout(20.0)
    self.world = self.client.load_world(params['town'])
    print('Carla server connected!')

    # Configure Traffic Manager settings
    self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
    self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    self.traffic_manager.set_synchronous_mode(self.sync_mode)
    self.traffic_manager.set_respawn_dormant_vehicles(True)
    self.traffic_manager.set_random_device_seed(0)

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor
    self.lidar_data = None
    self.lidar_height = 2.1
    self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '32')
    self.lidar_bp.set_attribute('range', '5000')
    self.lidar_bp.set_attribute('points_per_second', '100000')
    self.lidar_bp.set_attribute('rotation_frequency', '10')
    self.lidar_bp.set_attribute('upper_fov', '10')
    self.lidar_bp.set_attribute('lower_fov', '-30')

    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.synchronous_mode = self.sync_mode
    self.settings.fixed_delta_seconds = self.dt
    self.world.apply_settings(self.settings)

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0
    
    # Keep track of spawned actors
    self.spawned_vehicles = []
    self.spawned_walkers = []
    self.spawned_walker_controllers = []

    # Initialize pygame for visualization
    if self.rendering:
      pygame.init()
      pygame.font.init()
      self._init_renderer()

    # Initialize the sensor data lists
    self.sensor_list = []
    
    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T

  def reset(self, *, seed=None, options=None):
    if seed is not None:
        self.seed(seed)
    
    # Delete sensors, vehicles and walkers
    self._clear_all_actors()

    # Disable sync mode
    self.settings.synchronous_mode = False
    self.settings.no_rendering_mode = True

    self.world.apply_settings(self.settings)

    # Spawn surrounding vehicles
    self._spawn_surrounding_vehicles()
    
    # Spawn pedestrians
    self._spawn_pedestrians()

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      if self.task_mode == 'random':
        transform = random.choice(self.vehicle_spawn_points)
      elif self.task_mode == 'roundabout':
        #self.start = [52.1+np.random.uniform(-5,5), -4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
        transform = carla.Transform(carla.Location(x=98.93, y=-6.89, z=5), carla.Rotation(yaw=-179.14))
        
        #transform = set_carla_transform(self.start)
      
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: self._on_collision(event))
    self.sensor_list.append(self.collision_sensor)
    
    # Add lidar sensor
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    self.lidar_sensor.listen(lambda data: self._on_lidar_data(data))
    self.sensor_list.append(self.lidar_sensor)

    # Add camera sensor
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: self._on_camera_img(data))
    self.sensor_list.append(self.camera_sensor)

    # Update timesteps
    self.time_step = 0
    self.reset_step += 1

    # Enable sync mode
    self.settings.synchronous_mode = self.sync_mode
    self.world.apply_settings(self.settings)

    # Set up route planner
    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    
    # Initialize waypoints attribute before calling _get_obs()
    self.waypoints = []
    self.vehicle_front = False
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Set ego information for render
    if self.rendering:
      self.birdeye_render.set_hero(self.ego, self.ego.id)

    # Tick once to ensure all sensors data are available
    if self.sync_mode:
      self.world.tick()
    else:
      time.sleep(0.1)
    
    # Get the observation
    obs = self._get_obs()
    
    # For new Gym API
    info = {}
    return obs, info

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: self._on_collision(event))
    self.sensor_list.append(self.collision_sensor)
    
    # Add lidar sensor
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    self.lidar_sensor.listen(lambda data: self._on_lidar_data(data))
    self.sensor_list.append(self.lidar_sensor)

    # Add camera sensor
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: self._on_camera_img(data))
    self.sensor_list.append(self.camera_sensor)

    # Update timesteps
    self.time_step = 0
    self.reset_step += 1

    # Enable sync mode
    self.settings.synchronous_mode = self.sync_mode
    self.world.apply_settings(self.settings)

    # Set up route planner
    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Set ego information for render
    if self.rendering:
      self.birdeye_render.set_hero(self.ego, self.ego.id)

    # Tick once to ensure all sensors data are available
    if self.sync_mode:
      self.world.tick()
    else:
      time.sleep(0.1)
      
    return self._get_obs()
  
  def step(self, action):
    """Execute one step of the environment.
    
    Args:
        action: [acceleration, steering] for continuous mode or
                action index for discrete mode
    
    Returns:
        observation, reward, terminated, truncated, info
    """
    # Calculate acceleration and steering
    if self.discrete:
        acc = self.discrete_act[0][action//self.n_steer]
        steer = self.discrete_act[1][action%self.n_steer]
    else:
        acc = action[0]
        steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
        throttle = np.clip(acc/3, 0, 1)
        brake = 0
    else:
        throttle = 0
        brake = np.clip(-acc/8, 0, 1)

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    # Tick the world
    if self.sync_mode:
        self.world.tick()
    else:
        # Sleep a bit to ensure we get new sensor data
        time.sleep(0.1)

    # Update jaywalkers periodically
    self._update_jaywalkers()

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
        self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
        self.walker_polygons.pop(0)

    # Update route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Get observation, reward, and terminal state
    obs = self._get_obs()
    reward = self._get_reward()
    done = self._terminal()
    
    # Split 'done' into terminated and truncated
    terminated = done
    truncated = False
    # If we're just ending due to max steps, that's truncated not terminated
    if self.time_step > self.max_time_episode:
        terminated = False
        truncated = True
    
    # Update time step
    self.time_step += 1
    
    # Information dictionary
    info = {
        'waypoints': self.waypoints,
        'vehicle_front': self.vehicle_front
    }
    
    # Make sure observations are float32
    if 'state' in obs:
        obs['state'] = obs['state'].astype(np.float32)
    
    if self.time_step % 100 == 0:  # Print every 100 steps
      self.print_pedestrian_info()

    return obs, reward, terminated, truncated, info
  
  def close(self):
    """Clean up the environment"""
    self._clear_all_actors()
    if self.rendering:
      pygame.quit()
  
  def seed(self, seed=None):
    """Set random seed"""
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode='human'):
    """Render the environment"""
    if self.rendering:
        # Process Pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                
        if not self.sync_mode:
            self._get_obs()
        pygame.display.flip()
        time.sleep(0.01)  # Small delay to keep the UI responsive
 

  def _spawn_surrounding_vehicles(self):
    """Spawn surrounding vehicles"""
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

  def _spawn_pedestrians(self):
    """Spawn pedestrians with improved jaywalking behavior"""
    count = self.number_of_walkers
    
    # Check if jaywalking is enabled
    enable_jaywalking = hasattr(self, 'jaywalking_pedestrians') and self.jaywalking_pedestrians
    
    # Set walker percentages
    jaywalkers_percent = 0.7 if enable_jaywalking else 0.0
    
    # Track spawned pedestrians
    spawned_normal = 0
    spawned_jaywalkers = 0
    
    # For debugging: get ego vehicle position if available
    ego_pos = None
    if hasattr(self, 'ego') and self.ego:
        ego_pos = self.ego.get_location()
    
    # Clear existing spawn points and generate new ones near ego
    self.walker_spawn_points = []
    
    if ego_pos:
        # Generate spawn points near the ego vehicle for better visibility
        for _ in range(count * 2):  # Generate more points than needed
            spawn_point = carla.Transform()
            
            # Random angle and distance from ego
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.uniform(15, 50)  # 15-50 meters from ego
            
            # Calculate position
            x = ego_pos.x + distance * math.cos(angle)
            y = ego_pos.y + distance * math.sin(angle)
            
            # Get closest point on the sidewalk or road
            loc = carla.Location(x=x, y=y, z=ego_pos.z)
            waypoint = self.world.get_map().get_waypoint(loc, project_to_road=True)
            
            if waypoint:
                # Use waypoint location with slight offset for better spawning
                spawn_point.location = waypoint.transform.location
                spawn_point.location.z += 0.5  # Small elevation to avoid spawning inside the ground
                self.walker_spawn_points.append(spawn_point)
    
    # Add some random spawns from navigation for diversity
    for _ in range(count):
        spawn_point = carla.Transform()
        loc = self.world.get_random_location_from_navigation()
        if loc:
            spawn_point.location = loc
            self.walker_spawn_points.append(spawn_point)
    
    # Shuffle spawn points for random selection
    random.shuffle(self.walker_spawn_points)
    
    # Try to spawn pedestrians
    count_remaining = count
    for spawn_point in self.walker_spawn_points:
        # Determine if this walker should be a jaywalker
        is_jaywalker = enable_jaywalking and random.random() < jaywalkers_percent
        
        if self._try_spawn_random_walker_at(spawn_point, is_jaywalker=is_jaywalker):
            if is_jaywalker:
                spawned_jaywalkers += 1
            else:
                spawned_normal += 1
                
            count_remaining -= 1
            if count_remaining <= 0:
                break
    
    print(f"Spawned {spawned_normal} normal pedestrians and {spawned_jaywalkers} jaywalkers")


  def _on_collision(self, event):
    """Callback for collision sensor"""
    impulse = event.normal_impulse
    intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    self.collision_hist.append(intensity)
    if len(self.collision_hist) > self.collision_hist_l:
      self.collision_hist.pop(0)

  def _on_lidar_data(self, data):
    """Callback for lidar sensor"""
    self.lidar_data = data

  def _on_camera_img(self, data):
    """Callback for camera sensor"""
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    self.camera_img = array

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    if len(blueprint_library) == 0:
      # If no blueprint found with the specified number of wheels, use the first available
      blueprint_library = blueprints
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    if not self.rendering:
      return
      
    self.display = pygame.display.set_mode(
      (self.display_size * 3, self.display_size),
      pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random blueprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
        vehicle.set_autopilot(True, self.tm_port)
        self.spawned_vehicles.append(vehicle)  # Track the vehicle
        return True
    return False


  def _try_spawn_random_walker_at(self, transform, is_jaywalker=False):
    """Spawn a walker with improved behavior"""
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    if walker_bp.has_attribute('is_invincible'):
        walker_bp.set_attribute('is_invincible', 'false')
    
    # Add color attributes to distinguish walker types visually
    if walker_bp.has_attribute('color'):
        if is_jaywalker:
            walker_bp.set_attribute('color', '255,0,0')  # Red for jaywalkers
        else:
            walker_bp.set_attribute('color', '0,0,255')  # Blue for normal walkers
    
    # Try to spawn the walker
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)
    
    if walker_actor is not None:
        self.spawned_walkers.append(walker_actor)
        
        # Create and start the controller
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
        self.spawned_walker_controllers.append(walker_controller_actor)
        
        # Start the controller
        walker_controller_actor.start()
        
        # Set initial destination
        if is_jaywalker:
            # Store jaywalker info for later updates
            if not hasattr(self, '_jaywalkers'):
                self._jaywalkers = []
            self._jaywalkers.append((walker_actor, walker_controller_actor))
            
            # Set higher speed for jaywalkers
            walker_controller_actor.set_max_speed(1.8 + random.random())
            
            # Set initial destination to a random point
            self._set_jaywalker_destination(walker_actor, walker_controller_actor)
        else:
            # Normal pedestrian behavior - stay on sidewalks
            walker_controller_actor.set_max_speed(1.0 + random.random() * 0.5)
            loc = self.world.get_random_location_from_navigation()
            if loc:
                walker_controller_actor.go_to_location(loc)
        
        return True
    
    return False

  def _set_jaywalker_destination(self, walker, controller):
    """Set destination for jaywalker to cross roads"""
    # Only proceed if we have an ego vehicle
    if not hasattr(self, 'ego') or not self.ego:
        # Fallback to random destination
        loc = self.world.get_random_location_from_navigation()
        if loc:
            controller.go_to_location(loc)
        return
    
    try:
        # Get ego vehicle information
        ego_location = self.ego.get_location()
        ego_transform = self.ego.get_transform()
        
        # Get forward vector and speed
        forward_vector = ego_transform.get_forward_vector()
        ego_speed = self.ego.get_velocity()
        speed_scalar = math.sqrt(ego_speed.x**2 + ego_speed.y**2)
        
        # Calculate distance ahead based on ego speed (greater distance for higher speeds)
        distance_ahead = max(15, min(30, speed_scalar * 5))  # Between 15 and 30 meters
        
        # Calculate point ahead of vehicle
        point_ahead = ego_location + forward_vector * distance_ahead
        
        # Get right vector (perpendicular to forward vector)
        right_vector = carla.Vector3D(x=-forward_vector.y, y=forward_vector.x, z=0)
        
        # Determine crossing direction
        crossing_distance = random.uniform(5, 10)
        
        if random.random() < 0.5:
            # Cross from right to left
            start_point = point_ahead + right_vector * crossing_distance
            end_point = point_ahead - right_vector * crossing_distance
        else:
            # Cross from left to right
            start_point = point_ahead - right_vector * crossing_distance
            end_point = point_ahead + right_vector * crossing_distance
        
        # Current walker location
        walker_location = walker.get_location()
        
        # If walker is far from crossing start point, send to start first
        if walker_location.distance(start_point) > 2.0:
            controller.go_to_location(start_point)
        else:
            # Walker is at start point, cross the road
            controller.go_to_location(end_point)
        
        # Set speed based on urgency (jaywalkers sometimes rush)
        if random.random() < 0.3:  # 30% chance of rushing
            controller.set_max_speed(2.5 + random.random())  # Fast crossing
        else:
            controller.set_max_speed(1.4 + random.random() * 0.8)  # Normal crossing
            
    except Exception as e:
        print(f"Error setting jaywalker destination: {e}")
        # Fallback to random destination
        loc = self.world.get_random_location_from_navigation()
        if loc:
            controller.go_to_location(loc)


  def _setup_jaywalker(self, walker, controller):
    """
    Configure a walker to exhibit jaywalking behavior.
    
    Args:
        walker: walker actor
        controller: AI controller for the walker
    """
    # Start the controller
    controller.start()
    
    # Set a higher walking speed for jaywalkers
    controller.set_max_speed(1.8 + random.random())  # Faster speed between 1.8 and 2.8
    
    # Set up a timer to periodically change jaywalker destination
    self._schedule_jaywalker_crossing(walker, controller)

  def _schedule_jaywalker_crossing(self, walker, controller):
    """
    Schedule a jaywalker to cross the road near the ego vehicle.
    
    Args:
        walker: walker actor
        controller: AI controller for the walker
    """
    # This method will be called periodically to update jaywalker destination
    
    # Only proceed if walker and controller are still alive
    if not walker.is_alive() or not controller.is_alive():
        return
    
    try:
        # Get ego vehicle location
        if self.ego and self.ego.is_alive():
            ego_location = self.ego.get_location()
            ego_transform = self.ego.get_transform()
            ego_velocity = self.ego.get_velocity()
            
            # Calculate the forward vector of the ego vehicle
            forward_vector = ego_transform.get_forward_vector()
            
            # Calculate a point ahead of the vehicle (30-60 meters)
            distance_ahead = random.uniform(30, 60)
            point_ahead = ego_location + forward_vector * distance_ahead
            
            # Calculate a point perpendicular to the vehicle's path (for crossing)
            right_vector = carla.Vector3D(x=-forward_vector.y, y=forward_vector.x, z=0)
            
            # Get current location of the pedestrian
            walker_location = walker.get_location()
            
            # Determine which side of the road to start from (random)
            cross_distance = random.uniform(5, 15)  # Distance across road
            
            if random.random() < 0.5:
                # Cross from right to left
                start_point = point_ahead + right_vector * cross_distance
                end_point = point_ahead - right_vector * cross_distance
            else:
                # Cross from left to right
                start_point = point_ahead - right_vector * cross_distance
                end_point = point_ahead + right_vector * cross_distance
            
            # If the walker is not already near the start point, send them there first
            if walker_location.distance(start_point) > 2.0:
                controller.go_to_location(self.world.get_random_location_from_navigation())
            else:
                # The walker is in position, have them cross the road
                controller.go_to_location(end_point)
        else:
            # No ego vehicle, just walk randomly
            controller.go_to_location(self.world.get_random_location_from_navigation())
    
    except Exception as e:
        print(f"Error scheduling jaywalker crossing: {e}")
    
    # Schedule the next update after a random delay
    # In a real implementation, you would use a proper scheduler or the CARLA callback system
    # For this example, we're using a simple approach based on simulation steps
    self._jaywalker_update_counter = random.randint(50, 200)  # Update after 5-20 seconds assuming 10 Hz
    
  def _update_jaywalkers(self):
    """Update jaywalker behavior during simulation steps"""
    if not hasattr(self, '_jaywalkers'):
        self._jaywalkers = []
        return
        
    if not hasattr(self, '_jaywalker_update_counters'):
        self._jaywalker_update_counters = {}
    
    # Only update if we have the ego vehicle
    if not hasattr(self, 'ego') or not self.ego:
        return
    
    # Update counters and jaywalker destinations
    for i, (walker, controller) in enumerate(self._jaywalkers):
        # Skip if walker or controller is no longer valid
        if not walker or not controller:
            continue
            
        try:
            # Skip if not alive
            if not walker.is_alive or not controller.is_alive:
                continue
                
            # Initialize or update counter
            if i not in self._jaywalker_update_counters:
                self._jaywalker_update_counters[i] = random.randint(30, 100)  # Update more frequently
            else:
                self._jaywalker_update_counters[i] -= 1
            
            # Time to update this jaywalker
            if self._jaywalker_update_counters[i] <= 0:
                # Update destination
                self._set_jaywalker_destination(walker, controller)
                
                # Reset counter with random value
                self._jaywalker_update_counters[i] = random.randint(30, 100)
        except Exception as e:
            print(f"Error updating jaywalker {i}: {e}")

  def print_pedestrian_info(self):
    """Print debug information about pedestrians in the scene"""
    
    if not hasattr(self, 'ego') or not self.ego:
        print("No ego vehicle to reference")
        return
        
    ego_location = self.ego.get_location()
    
    print("\n---- Pedestrian Debug Info ----")
    print(f"Ego vehicle at: ({ego_location.x:.1f}, {ego_location.y:.1f}, {ego_location.z:.1f})")
    print(f"Total pedestrians: {len(self.spawned_walkers)}")
    
    # Count pedestrians within different distance ranges
    close_range = 0  # Within 20m
    mid_range = 0    # 20-50m
    far_range = 0    # >50m
    
    for i, walker in enumerate(self.spawned_walkers):
        try:
            if walker and hasattr(walker, 'is_alive') and walker.is_alive:
                walker_loc = walker.get_location()
                distance = walker_loc.distance(ego_location)
                
                # Count by distance
                if distance < 20:
                    close_range += 1
                elif distance < 50:
                    mid_range += 1
                else:
                    far_range += 1
                
                # Check if this is a jaywalker
                is_jaywalker = False
                if hasattr(self, '_jaywalkers'):
                    for j_walker, _ in self._jaywalkers:
                        if j_walker.id == walker.id:
                            is_jaywalker = True
                            break
                
                # Only print detailed info for close pedestrians to avoid cluttering the output
                if distance < 30:  # Only show details for nearby walkers
                    print(f"Walker {i}: {'Jaywalker' if is_jaywalker else 'Normal'}")
                    print(f"  - Position: ({walker_loc.x:.1f}, {walker_loc.y:.1f}, {walker_loc.z:.1f})")
                    print(f"  - Distance to ego: {distance:.1f}m")
        except Exception as e:
            print(f"Error processing walker {i}: {str(e)}")
    
    # Print summary by distance
    print(f"Pedestrian distance summary:")
    print(f"  - Close range (<20m): {close_range}")
    print(f"  - Mid range (20-50m): {mid_range}")
    print(f"  - Far range (>50m): {far_range}")
    print("--------------------------------\n")


  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break
    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego = vehicle
      return True
      
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans = actor.get_transform()
      x = trans.location.x
      y = trans.location.y
      yaw = trans.rotation.yaw/180*np.pi
      # Get length and width
      bb = actor.bounding_box
      l = bb.extent.x
      w = bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local = np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly = np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id] = poly
    return actor_poly_dict

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    if self.rendering:
      self.birdeye_render.vehicle_polygons = self.vehicle_polygons
      self.birdeye_render.walker_polygons = self.walker_polygons
      self.birdeye_render.waypoints = self.waypoints

      # birdeye view with roadmap and actors
      birdeye_render_types = ['roadmap', 'actors']
      if self.display_route:
        birdeye_render_types.append('waypoints')
      self.birdeye_render.render(self.display, birdeye_render_types)
      birdeye = pygame.surfarray.array3d(self.display)
      birdeye = birdeye[0:self.display_size, :, :]
      birdeye = display_to_rgb(birdeye, self.obs_size)

      # Roadmap
      if self.pixor:
        roadmap_render_types = ['roadmap']
        if self.display_route:
          roadmap_render_types.append('waypoints')
        self.birdeye_render.render(self.display, roadmap_render_types)
        roadmap = pygame.surfarray.array3d(self.display)
        roadmap = roadmap[0:self.display_size, :, :]
        roadmap = display_to_rgb(roadmap, self.obs_size)
        # Add ego vehicle
        for i in range(self.obs_size):
          for j in range(self.obs_size):
            if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
              roadmap[i, j, :] = birdeye[i, j, :]

      # Display birdeye image
      birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
      self.display.blit(birdeye_surface, (0, 0))

      ## Lidar image generation
      point_cloud = []
      # Get point cloud data
      if self.lidar_data is not None:
        for detection in self.lidar_data:
          point_cloud.append([detection.point.x, detection.point.y, -detection.point.z])
        point_cloud = np.array(point_cloud)
        # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
        # and z is set to be two bins.
        y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
        x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
        z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
        # Get lidar image according to the bins
        lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
        lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
        # Add the waypoints to lidar image
        if self.display_route:
          wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
        else:
          wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
        wayptimg = np.expand_dims(wayptimg, axis=2)
        wayptimg = np.fliplr(np.rot90(wayptimg, 3))

        # Get the final lidar image
        lidar = np.concatenate((lidar, wayptimg), axis=2)
        lidar = np.flip(lidar, axis=1)
        lidar = np.rot90(lidar, 1)
        lidar = lidar * 255

        # Display lidar image
        lidar_surface = rgb_to_display_surface(lidar, self.display_size)
        self.display.blit(lidar_surface, (self.display_size, 0))

        ## Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size * 2, 0))

        # Display on pygame
        pygame.display.flip()
      else:
        # When lidar data is not available, create empty arrays
        lidar = np.zeros((self.obs_size, self.obs_size, 3))
    else:
      # Create empty arrays when rendering is disabled
      birdeye = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
      lidar = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
      if self.pixor:
        roadmap = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        
    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front], dtype=np.float32)

    if self.pixor:
      ## Vehicle classification and regression maps (requires further normalization)
      vh_clas = np.zeros((self.pixor_size, self.pixor_size))
      vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

      # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
      # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
      for actor in self.world.get_actors().filter('vehicle.*'):
        x, y, yaw, l, w = get_info(actor)
        x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
        if actor.id != self.ego.id:
          if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
            x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
              local_info=(x_local, y_local, yaw_local, l, w),
              d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
            cos_t = np.cos(yaw_pixel)
            sin_t = np.sin(yaw_pixel)
            logw = np.log(w_pixel)
            logl = np.log(l_pixel)
            pixels = get_pixels_inside_vehicle(
              pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
              pixel_grid=self.pixel_grid)
            for pixel in pixels:
              vh_clas[pixel[0], pixel[1]] = 1
              dx = x_pixel - pixel[0]
              dy = y_pixel - pixel[1]
              vh_regr[pixel[0], pixel[1], :] = np.array(
                [cos_t, sin_t, dx, dy, logw, logl])

      # Flip the image matrix so that the origin is at the left-bottom
      vh_clas = np.flip(vh_clas, axis=0)
      vh_regr = np.flip(vh_regr, axis=0)

      # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
      pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]
      
    obs = {
        'camera': camera.astype(np.uint8),
        'lidar': lidar.astype(np.uint8),
        'birdeye': birdeye.astype(np.uint8),
        'state': state,  # Already float32
    }

    if self.pixor:
      obs.update({
        'roadmap': roadmap.astype(np.uint8),
        'vh_clas': np.expand_dims(vh_clas, -1).astype(np.float32),
        'vh_regr': vh_regr.astype(np.float32),
        'pixor_state': pixor_state,
      })

    return obs
    
  def _get_reward(self):
    """Calculate the step reward."""
    # Get ego vehicle state
    ego_transform = self.ego.get_transform()
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    speed_kmh = speed * 3.6
    ego_x, ego_y = get_pos(self.ego)

    # 1. Speed reward - incentivize speeds up to target_speed_kmh
    r_speed = - (5 ** (abs(speed_kmh - self.target_speed_kmh)/10)) / (5 ** 3)
    '''
    if speed_kmh <= self.target_speed_kmh:
        # Proportional to speed up to target (max reward of 2.0 at target speed)
        r_speed = 2.0 * (speed_kmh / self.target_speed_kmh)
    else:
        # Penalize excess speed (negative reward increases with speed over target)
        excess = speed_kmh - self.target_speed_kmh
        r_speed = -1.0 * (excess / 10.0)  # -0.1 per km/h over limit
    '''
    # Find the 3rd nearest waypoint in front of the vehicle
    target_waypoint = find_target_waypoint(self.waypoints, ego_transform, n_th_nearest=3)

    '''
    # 2. Distance to target waypoint
    r_waypoint_distance = -10
    if target_waypoint is not None:
        dist = calculate_distance_to_waypoint(ego_x, ego_y, target_waypoint)
        
        if dist < self.waypoint_target_distance:
            # Positive reward if close (max reward of 3.0 when at the waypoint)
            r_waypoint_distance = 3.0 * ((self.waypoint_target_distance - dist) / self.waypoint_target_distance)
        else:
            # Negative reward if far (increases with distance)
            excess_dist = dist - self.waypoint_target_distance
            r_waypoint_distance = -0.5 * (excess_dist / 10.0)  # -0.05 per meter over threshold
    '''

    # 3. Heading alignment with the vector to the target waypoint
    r_heading =-1
    if target_waypoint is not None:
        # Calculate alignment (-1 to 1)
        alignment = calculate_heading_alignment(ego_transform, target_waypoint)
        
        # Scale to reward (max 2.0 for perfect alignment)
        r_heading = alignment
    
    # Collision penalty (most severe)
    r_collision = 0
    if len(self.collision_hist) > 0:
        r_collision = -3 # Very large negative reward
    
    # Waypoint deviation penalty (for large deviations, but not termination yet)
    r_deviation = 0
    if len(self.waypoints) > 0:
        min_dist = float('inf')
        for waypoint in self.waypoints:
            dist = calculate_distance_to_waypoint(ego_x, ego_y, waypoint)
            min_dist = min(min_dist, dist)
        #print("min dist", min_dist)
        r_deviation = (5 - min_dist)/5
        if r_deviation < -1:
           r_deviation = -1
        '''
        r_deviation = self.waypoint_distance_threshold / min_dist
        if min_dist > waypoint_warning_threshold:
            #deviation_factor = (min_dist - waypoint_warning_threshold) / (self.waypoint_distance_threshold - waypoint_warning_threshold)
            r_deviation = -min_dist  # Up to -50 as it approaches termination threshold
        if min_dist > self.waypoint_distance_threshold:
            r_deviation = - 50'''
    
    # Combine all reward components with weights
    #print("collision ", r_collision, "  speed  ", r_speed, " r_deviation  ", r_deviation, "  r_heading  ", r_heading)
    r = r_collision + (0.4 * r_speed) + (0.25 * r_deviation) + (0.35 * r_heading) # this -0.25 because overcome this issue if the agent wait the final total reward dosn't increase
    #print("r ", r)
    
    return r

    '''
    # Reward for speed tracking
    r_speed = -abs(speed - self.desired_speed)
    
    # Collision penalty (most severe)
    r_collision = 0
    if len(self.collision_hist) > 0:
        r_collision = -100  # Very large negative reward

    # Reward for steering (penalize excessive steering)
    r_steer = -self.ego.get_control().steer**2

    # Lane departure penalty (moderate severity)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
        r_out = -10  # Moderate negative reward
    
    # Calculate longitudinal speed (speed along the lane direction)
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # Penalty for excessive speed
    r_fast = 0
    if lspeed_lon > self.desired_speed:
        r_fast = -1
    
    # Penalty for lateral acceleration (discourage hard turns at high speed)
    r_lat = -abs(self.ego.get_control().steer) * lspeed_lon**2
    
    # Destination reached bonus (large positive reward)
    r_destination = 0
    if self.dests is not None:
        for dest in self.dests:
            if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 4:
                r_destination = 100  # Large positive reward
                break
    
    # Combine all reward components
    r = r_collision + r_destination + 1*lspeed_lon + 10*r_fast + r_out*10 + r_steer*5 + 0.2*r_lat - 0.1
    
    # Add speed bonus to encourage movement
    speed_bonus = min(speed / 5.0, 1.0)  # Bonus for speed up to 5 m/s
    '''
    return r + speed_bonus

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides (always terminate)
    if len(self.collision_hist) > 0: 
        return True

    # If reach maximum timestep
    if self.time_step > self.max_time_episode:
        return True

    # If at destination
    if self.dests is not None:
        for dest in self.dests:
            if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 5:
                return True

    # If out of lane (with configurable behavior)
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if self.terminate_on_lane_departure and abs(dis) > self.out_lane_thres * self.lane_departure_factor:
        return True

    return False
    
  def _clear_all_actors(self):
    """Clear actors with better error handling."""
    # Stop all walker controllers
    for controller in self.spawned_walker_controllers:
        if controller and controller.is_alive:
            try:
                controller.stop()
            except:
                pass
                
    # Destroy controllers
    for controller in self.spawned_walker_controllers:
        if controller and controller.is_alive:
            try:
                controller.destroy()
            except:
                pass
    self.spawned_walker_controllers = []
    
    # Destroy walkers
    for walker in self.spawned_walkers:
        if walker and walker.is_alive:
            try:
                walker.destroy()
            except:
                pass
    self.spawned_walkers = []
    
    # Destroy vehicles
    for vehicle in self.spawned_vehicles:
        if vehicle and vehicle.is_alive:
            try:
                vehicle.destroy() 
            except:
                pass
    self.spawned_vehicles = []
    
    # Destroy sensors
    for sensor in self.sensor_list:
        if sensor and sensor.is_alive:
            try:
                sensor.destroy()
            except:
                pass
    self.sensor_list = []