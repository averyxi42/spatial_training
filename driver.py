import ray
import os
import sys
import platform

# Get the IP from environment or hardcode it
head_ip = os.environ.get("HEAD_IP")

if not head_ip:
    print("Error: HEAD_IP environment variable not set.")
    sys.exit(1)

print(f"Connecting to Ray Cluster at {head_ip}:6379...")

# Connect to the existing cluster
ray.init(address=f"{head_ip}")

print("Connected!")
print(f"Cluster Resources: {ray.cluster_resources()}")

# --- Define Tasks for Each Environment ---

# Resource 'env_a' was defined in your Head start command
@ray.remote(resources={"env_a": 0.5})
def task_a():
    import sys
    return f"Task A running on Python {sys.version.split()[0]} (Container A)"

# Resource 'env_b' was defined in your Worker start command
@ray.remote(resources={"env_b": 0.5})
def task_b():
    import sys
    return f"Task B running on Python {sys.version.split()[0]} (Container B)"

# --- Run ---
print("\nSubmitting tasks...")
ref_a = task_a.remote()
ref_b = task_b.remote()

results = ray.get([ref_a, ref_b])

print("\n--- Results ---")
for res in results:
    print(res)