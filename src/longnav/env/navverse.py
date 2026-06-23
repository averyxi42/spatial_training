

import numbers
import torch
class NavverseEnv():
    def __init__(self): 
        import os

        def is_linux_headless():
            # Returns True if no display environment variable is set
            has_x11 = "DISPLAY" in os.environ
            has_wayland = "WAYLAND_DISPLAY" in os.environ
            return not (has_x11 or has_wayland)

        import argparse
        import json
        import numpy as np
        import torch
        import time

        # start simulation
        from isaaclab.app import AppLauncher
        import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
        import navverse.vln_args as vln_cli_args
        # Add command line arguments
        parser = argparse.ArgumentParser(description="Benchmark")
        rsl_rl_cli_args.add_rsl_rl_args(parser)
        vln_cli_args.add_vln_args(parser)
        AppLauncher.add_app_launcher_args(parser)
        import sys

        # 1. Back up the original arguments
        original_argv = sys.argv.copy()

        # 2. Clear sys.argv (keeps the script name at index 0)
        sys.argv = [sys.argv[0]]

        # -- Your code runs here with cleared arguments --
        args = vln_cli_args.parse_args(parser)

        # 3. Restore the original arguments when done
        sys.argv = original_argv
        args.disable_socket_server=True
        args.episode_folder = "/home/huyu/Documents/code/NavVerse-Benchmark/episodes"
        args.scene_folder = "/home/huyu/navverse_data/"
        args.headless = is_linux_headless()
        # Launch Isaac Lab app
        sim_start_time = time.time()
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        # Enable Extension and setup settings
        import carb
        import omni.kit.app
        # from isaacsim.core.utils.extensions import enable_extension
        # enable_extension("omni.anim.navigation.bundle")
        settings = carb.settings.get_settings()
        with open("settings.json", "w+") as f:
            json.dump(settings.get_settings_dictionary().get_dict(), f, indent=4)
        # settings.set("/renderer/multiGPU/enabled", False)
        # settings.set("/renderer/activeGpu", 0) 

        omni.kit.app.get_app().update()
        import isaaclab.sim as sim_utils

        # Local imports
        from navverse.sim import VLNSim

        # setup simulation
        self.vln_sim = VLNSim(args)
        # vln_sim.load_episode("test_generator_0")
        # vln_sim._reset()
        self.action_map = [
            [[0,0,0]],
            [[1,0,0]],
            [[0,0,np.deg2rad(30)]],
            [[0,0,-np.deg2rad(30)]]
        ]
        self.episode_ptr=0
        self.episodes = []
        # self.assign_shard(['test_generator_0'
        # ])
   
    def assign_shard(self, episodes: list[str]|None = None):
        '''
        assign a list of episodes identified via strings to the actor.
        if None is passed, load all available episodes.
        '''
        self.episodes = episodes
        self.episode_ptr = 0
        
    def flush_logs_to_disk(self):
        '''
        flush any internal logging. returns either None or a path pointing to a json file.
        '''
        pass
    
    def is_exhausted(self):
        '''
        returns True if the actor has exhausted its assigned episodes.
        '''
        return self.episode_ptr>=len(self.episodes)
    
    def reset(self):
        obs,info = self.vln_sim.reset_sync(self.episodes[self.episode_ptr])
        self.episode_ptr+=1
        return self._convert((obs,0,False,info))
    
    def _step(self,action):
        if action == 0:
            with torch.inference_mode():
                self.vln_sim.env.set_stop_called(0,True)
                step_data = self.vln_sim.env.step([0,0,0])
        else:
            step_data = self.vln_sim.step_relative_waypoints_sync(self.action_map[action])
        return step_data
    
    def _convert(self,state_tuple):
        obs,reward,done,info = state_tuple
        rgb = obs['pov_rgb'].squeeze().cpu().numpy()
        state = {
            "obs":{
                "instr_or_goal":self.vln_sim.current_episode['objnav'] #instruction
            },
            "reward":reward.squeeze().item() if not isinstance(reward, numbers.Number) else reward,
            "done":done.item() if not isinstance(done,bool) else done,
            "info":info['measurements'],
            "is_exhausted":self.is_exhausted()
        }
        return rgb,state
    
    def step(self,action, supplementary_logs):
        return self._convert(self._step(action))