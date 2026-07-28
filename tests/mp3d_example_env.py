from longnav.env.habitat import HabitatWorker
worker = HabitatWorker(config_path="habitat_configs/objectnav_mp3d_test.yaml")
from habitat import make_dataset 
test_dataset = make_dataset("ObjectNav-v1", config=worker.config_env.habitat.dataset)