from typing import Any, Dict
import wandb
import numpy as np
import os
import subprocess

class WandbLoggerActor:
    def __init__(self, wandb_init_kwargs, run_config=None, log_raw=False,
                 commit_interval=5, compact_metrics=False):
        """
        Args:
            wandb_init_kwargs: Dict for wandb.init (project, entity, name).
            run_config: The main experiment config dict (hyperparameters).
            log_raw: Boolean toggle for raw data in tables.
            commit_interval: Frequency of table uploads.
        """
        # 1. Gather Environment Metadata (SLURM, Git)
        system_metadata = self._capture_system_metadata()
        
        # 2. Merge User Config with System Metadata
        # This ensures we don't overwrite user config, but append system info
        full_config = run_config if run_config else {}
        full_config.update(system_metadata)

        # 3. Initialize WandB
        # We pass the merged config here so it appears in the "Overview" tab
        self.run = wandb.init(
            **wandb_init_kwargs, 
            config=full_config, 
            reinit=True
        )
        
        self.log_raw = log_raw
        self.commit_interval = commit_interval
        self.compact_metrics = compact_metrics
        
        # Table State
        self.table = None
        self.columns = None
        self.rows_since_last_commit = 0
        self.eval_videos = []
        self.context_cycle = None
        self.context_phase = None

        self.defined_metrics = set()
        if self.compact_metrics:
            self.run.define_metric("cycle")
            for namespace in ("rollout", "train", "policy", "eval", "runtime"):
                self.run.define_metric(f"{namespace}/*", step_metric="cycle")

    def set_context(self, cycle: int, phase: str):
        """Attach a stable cycle and phase to subsequently logged episode rows."""
        self.context_cycle = int(cycle)
        self.context_phase = str(phase)

    def log_global_metrics(self, metrics: dict, step=None):
        """
        For Driver-side metrics: Training Loss, Learning Rate, Epoch, etc.
        """
        if step is not None:
            # If step is provided, we align the metric to that X-axis
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)
    def log(self,row):
        """
        Processes a single episode row with namespace-based media detection.
        """
        # --- 4. Log Live Scalars ---
        # Filter out heavy objects (Lists, Arrays, and the new Media objects)
        # This keeps the "Run Overview" page clean and fast.
        log_payload = {
            k: v for k, v in row.items() 
            if not isinstance(v, (list, np.ndarray, wandb.Image, wandb.Video))
        }
        for k, v in log_payload.items():
            if k not in self.defined_metrics:
                # Filter for numeric types (exclude bools, strings, and None)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    self.run.define_metric(k, summary="mean")
                    self.run.define_metric(k, summary="max")
                    self.run.define_metric(k, summary="min")
                    self.defined_metrics.add(k)
        self.run.log(log_payload)

    def _process_episode_row(self, row: Dict[str, Any]) -> dict:
        processed_row = {}
        for k, v in row.items():
            if k.startswith('raw/') and not self.log_raw:
                continue
            if k.startswith('img/') and v is not None:
                processed_row[k] = wandb.Image(v)
            elif k.startswith('vid/') and v is not None:
                processed_row[k] = wandb.Video(v, format="mp4")
            else:
                processed_row[k] = v
        return processed_row

    def log_row(self, row:Dict[str,Any]):
        """Processes a single episode row with namespace-based media detection."""
        if self.compact_metrics:
            if self.context_phase == "eval":
                video_path = row.get("vid/episode_video")
                if video_path is not None:
                    episode_id = str(
                        row.get("episode_label")
                        or row.get("eval_env/episode_label")
                        or ""
                    )
                    if not episode_id:
                        raise ValueError("eval video is missing its global episode ID")
                    self.eval_videos.append(
                        (
                            episode_id,
                            wandb.Video(
                                video_path,
                                caption=(
                                    f"step={self.context_cycle} "
                                    f"episode={episode_id}"
                                ),
                                format="mp4",
                            ),
                        )
                    )
            return

        processed_row = self._process_episode_row(row)

        # --- 2. Lazy Table Init ---
        if self.table is None:
            self.columns = sorted(list(processed_row.keys()))
            self.table = wandb.Table(columns=self.columns,allow_mixed_types=True,log_mode='MUTABLE')

        # --- 3. Add to Buffer ---
        # Use .get() to handle potential schema drift (though rare in this setup)
        table_row = [processed_row.get(c, None) for c in self.columns]
        self.table.add_data(*table_row)
        self.rows_since_last_commit += 1

        # --- 4. Log Live Scalars ---
        # Filter out heavy objects (Lists, Arrays, and the new Media objects)
        # This keeps the "Run Overview" page clean and fast.
        log_payload = {
            k: v for k, v in processed_row.items() 
            if not isinstance(v, (list, np.ndarray, wandb.Image, wandb.Video))
        }
        for k, v in log_payload.items():
            if k not in self.defined_metrics:
                # Filter for numeric types (exclude bools, strings, and None)
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    self.run.define_metric(k, summary="mean")
                    self.run.define_metric(k, summary="max")
                    self.run.define_metric(k, summary="min")
                    self.defined_metrics.add(k)

        # --- 5. Conditional Commit ---
        if self.rows_since_last_commit >= self.commit_interval:
            log_payload["episode_details"] = self.table
            # Reset table to prevent O(N^2) slowdown and race conditions
            # self.table = wandb.Table(columns=self.columns, allow_mixed_types=True)
            self.rows_since_last_commit = 0 
        self.run.log(log_payload)

    def _pop_eval_videos(self):
        videos, self.eval_videos = self.eval_videos, []
        return [video for _, video in sorted(videos, key=lambda item: item[0])]

    def log_cycle_metrics(self, metrics: dict, cycle: int, phase: str = "train"):
        """Write one compact history row for a completed phase."""
        payload = {"cycle": int(cycle), **metrics}
        self.run.log(payload)
        self.context_cycle = int(cycle)
        self.context_phase = str(phase)

    def log_eval_metrics(self, metrics: dict, cycle: int):
        payload = {"cycle": int(cycle), **metrics}
        if self.compact_metrics:
            videos = self._pop_eval_videos()
            if videos:
                payload["eval/video"] = videos
        self.run.log(payload)
        self.context_cycle = int(cycle)
        self.context_phase = "eval"

    def _capture_system_metadata(self):
        """
        Internal helper to grab SLURM and Git info.
        """
        meta = {}
        
        # A. SLURM Environment Variables
        # These are standard across most SLURM clusters
        slurm_keys = [
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
            "SLURM_NTASKS", "SLURM_CPUS_PER_TASK"
        ]
        for k in slurm_keys:
            if k in os.environ:
                meta[f"system/{k.lower()}"] = os.environ[k]

        # B. Git Commit Hash (Reproducibility)
        # We try to run git rev-parse; if it fails (not a repo), we skip.
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).strip().decode('utf-8')
            meta["system/git_commit"] = commit_hash
        except (OSError, subprocess.SubprocessError):
            pass

        return meta

    def alert(self, 
        title: str,
        text: str,
        level):
        """
        Error logging.
        Args:
            message: The error string.
            level: ERROR, INFO, WARN
            alert: If True, triggers a WandB Alert (useful for cluster failures).
        """       
        self.run.alert(
            title=title, 
            text=text, 
            level=level
        )  
        
    def flush(self):
        """Forces pending media or table buffers to WandB."""
        if self.compact_metrics:
            videos = self._pop_eval_videos()
            if videos:
                self.run.log({"cycle": self.context_cycle, "eval/video": videos})
            return
        if self.rows_since_last_commit > 0 and self.table is not None:
            # Create a payload with just the table
            payload = {"episode_details": self.table}
            self.run.log(payload)
            
            # Reset the counter
            self.rows_since_last_commit = 0
    def close(self):
        # Flush the table one last time
        if self.compact_metrics:
            self.flush()
        elif self.table and self.rows_since_last_commit > 0:
            self.run.log({"episode_details": self.table})
        self.run.finish()
