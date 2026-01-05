import os
import subprocess
import time
import signal
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from huggingface_hub import HfApi, CommitOperationAdd

# --- CONFIGURATION ---
DATA_ROOT = "/scratch/mcity_project_root/mcity_project/huyu/qixin/dump/habitat_web_image_depth"
REPO_ID = "Aasdfip/habitat_web_image_depth_RESCUE"
REPO_TYPE = "dataset"

TAR_WORKERS = 5 
BATCH_SIZE = 50 
STAGING_DIR = "/scratch/mcity_project_root/mcity_project/huyu/qixin/dump/staging_tars"

# Global exit flag for interrupts
EXIT_EVENT = threading.Event()

def signal_handler(sig, frame):
    # This triggers the graceful shutdown across threads and processes
    if not EXIT_EVENT.is_set():
        print("\n[!] Emergency Stop: Cleaning up and exiting...")
        EXIT_EVENT.set()

signal.signal(signal.SIGINT, signal_handler)

def get_existing_files(api):
    print(f"Checking existing files in {REPO_ID} to resume...")
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        return set(files)
    except Exception as e:
        print(f"Note: Repo might be new or empty: {e}")
        return set()

def tar_worker(episode_full_path):
    """
    Step 1: Create the tar file using an atomic 'write-then-rename' strategy.
    Deletes the temporary file on any error or interrupt.
    """
    # Quick exit if shutdown initiated
    if EXIT_EVENT.is_set(): return None
    
    scene_dir = os.path.dirname(episode_full_path)
    scene_name = os.path.basename(scene_dir)
    ep_name = os.path.basename(episode_full_path)
    
    tar_name = f"{scene_name}_{ep_name}.tar"
    local_final_path = os.path.join(STAGING_DIR, tar_name)
    local_tmp_path = local_final_path + ".tmp"
    remote_path = f"data/{scene_name}/{ep_name}.tar"

    # Resume check for staging area
    if os.path.exists(local_final_path):
        return (local_final_path, remote_path)

    try:
        # Create tar to a .tmp file
        # -C allows us to tar the episode folder without the full absolute path prefix
        subprocess.run(
            ["tar", "-cf", local_tmp_path, "-C", scene_dir, ep_name],
            check=True, capture_output=True
        )
        
        # Atomic rename: if this happens, the tar is definitely complete
        os.rename(local_tmp_path, local_final_path)
        return (local_final_path, remote_path)

    except (Exception, KeyboardInterrupt):
        # Clean up partial/corrupted files immediately
        if os.path.exists(local_tmp_path):
            try: os.remove(local_tmp_path)
            except: pass
        return None

def upload_manager(upload_queue, api):
    """
    Background Thread: Consumes tars from queue and commits in batches.
    """
    batch = []
    processed_count = 0

    # Continue until exit is set AND queue is empty
    while not EXIT_EVENT.is_set() or not upload_queue.empty():
        try:
            op_data = upload_queue.get(timeout=1)
            if op_data:
                local, remote = op_data
                batch.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local))
            
            # Commit if batch is full OR if we are shutting down and have leftovers
            if len(batch) >= BATCH_SIZE or (EXIT_EVENT.is_set() and batch):
                _execute_batch(api, batch)
                processed_count += len(batch)
                print(f"[*] Total Uploaded: {processed_count} files.")
                batch = []
                
        except queue.Empty:
            continue

    # Final sweep
    if batch:
        _execute_batch(api, batch)

def _execute_batch(api, batch):
    """Performs the HF API commit with exponential backoff."""
    for attempt in range(5):
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=batch,
                commit_message=f"Rescue batch: {len(batch)} tars"
            )
            # Cleanup local tars ONLY after the commit is successfully recorded on HF
            for op in batch:
                if os.path.exists(op.path_or_fileobj):
                    os.remove(op.path_or_fileobj)
            return
        except Exception as e:
            if "429" in str(e) or "500" in str(e):
                wait = (2 ** attempt)
                print(f"[!] Rate limited/Server Error. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[!] Critical Batch Error: {e}")
                break

def main():
    api = HfApi()
    os.makedirs(STAGING_DIR, exist_ok=True)
    
    # Ensure repo exists
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)

    # 1. Resume Check: What is already on HF?
    existing_remote_files = get_existing_files(api)
    
    # 2. Build Work Queue
    tasks = []
    skips = 0
    for scene in sorted(os.listdir(DATA_ROOT)):
        s_path = os.path.join(DATA_ROOT, scene)
        if not os.path.isdir(s_path): continue
        for ep in os.listdir(s_path):
            ep_path = os.path.join(s_path, ep)
            if os.path.isdir(ep_path):
                remote_target = f"data/{scene}/{ep}.tar"
                if remote_target not in existing_remote_files:
                    tasks.append(ep_path)
                else:
                    skips+=1

    print(f"Found {len(tasks)} files needing upload. skipped {skips} existing ones.")
    import random
    print("shuffling tasks")
    random.shuffle(tasks)
    print("done shuffling")
    if not tasks:
        print("Everything up to date.")
        return

    # 3. Start Upload Worker (Thread)
    upload_q = queue.Queue()
    up_thread = threading.Thread(target=upload_manager, args=(upload_q, api))
    up_thread.daemon = True
    up_thread.start()

    # 4. Start Tar Workers (Processes)
    print(f"Deploying {TAR_WORKERS} tar workers. Rescuing data...")
    with ProcessPoolExecutor(max_workers=TAR_WORKERS) as executor:
        futures = {executor.submit(tar_worker, t): t for t in tasks}
        
        try:
            for future in as_completed(futures):
                if EXIT_EVENT.is_set(): break
                result = future.result()
                if result:
                    upload_q.put(result)
        except Exception as e:
            print(f"Pipeline error: {e}")
        finally:
            # Signal the upload thread that no more tars are coming
            EXIT_EVENT.set()

    # Wait for the upload thread to finish the current/final batch
    up_thread.join(timeout=60)
    print("Rescue Operation Ended.")

if __name__ == "__main__":
    main()