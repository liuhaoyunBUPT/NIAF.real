from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class Debugger:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.step = 0
        self.frame: Optional[Dict[str, Any]] = None

        self._raw_actions: Optional[np.ndarray] = None
        self._post_actions: Optional[np.ndarray] = None
        self._raw_velocities: Optional[np.ndarray] = None
        self._post_velocities: Optional[np.ndarray] = None

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.output_dir = os.path.join(project_root, "debug")
        self.assets_dir = os.path.join(self.output_dir, "assets")
        self.latest_json = os.path.join(self.output_dir, "latest.json")

        if self.enabled:
            os.makedirs(self.assets_dir, exist_ok=True)
            self._write_dashboard_html()

    def begin_frame(
        self,
        *,
        obs: Dict[str, Any],
        model_name: str,
        action_mode: str,
        arm_mode: str,
        chunk_size: int,
        target_chunk_size: Optional[int],
        active_cameras: list[str],
        default_prompt: str,
    ) -> None:
        if not self.enabled:
            return

        self.step += 1
        self._raw_actions = None
        self._post_actions = None
        self._raw_velocities = None
        self._post_velocities = None

        images = obs.get("images", {})
        frame: Dict[str, Any] = {
            "step": self.step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "action_mode": action_mode,
            "arm_mode": arm_mode,
            "chunk_size": int(chunk_size),
            "target_chunk_size": target_chunk_size,
            "active_cameras": list(active_cameras),
            "raw_prompt": obs.get("prompt", default_prompt),
            "formatted_prompt": "",
            "cameras": [],
            "left_arm_plot_path": "",
            "right_arm_plot_path": "",
        }

        cam_map = {
            "rgb_static": "cam_high",
            "rgb_left_wrist": "cam_left_wrist",
            "rgb_right_wrist": "cam_right_wrist",
        }
        for cam_name in active_cameras:
            client_key = cam_map.get(cam_name)
            raw_img = images.get(client_key)
            raw_stats = self._array_stats(raw_img)
            raw_img_rel = ""
            if raw_img is not None:
                raw_img_rel = self._save_image(raw_img, f"step{self.step:06d}_{cam_name}_raw.png")
            frame["cameras"].append(
                {
                    "camera_name": cam_name,
                    "client_key": client_key,
                    "raw": raw_stats,
                    "preprocessed": {},
                    "raw_image_path": raw_img_rel,
                    "preprocessed_image_path": "",
                }
            )

        self.frame = frame

    def record_text(self, *, raw_prompt: str, formatted_prompt: str) -> None:
        if not self.enabled or self.frame is None:
            return
        self.frame["raw_prompt"] = raw_prompt
        self.frame["formatted_prompt"] = formatted_prompt

    def record_preprocessed_images(self, model_obs: Dict[str, Any]) -> None:
        if not self.enabled or self.frame is None:
            return

        rgb_obs = model_obs.get("rgb_obs", {})
        for cam in self.frame.get("cameras", []):
            cam_name = cam.get("camera_name", "")
            prep = rgb_obs.get(cam_name)
            cam["preprocessed"] = self._array_stats(prep)
            if prep is not None:
                cam["preprocessed_image_path"] = self._save_image(
                    prep,
                    f"step{self.step:06d}_{cam_name}_preprocessed.png",
                    is_tensor=True,
                )

    def record_actions(self, *, stage: str, actions: np.ndarray, arm_mode: str) -> None:
        if not self.enabled or self.frame is None:
            return
        if stage == "raw_model_output":
            self._raw_actions = np.asarray(actions, dtype=np.float32)
        elif stage == "postprocessed_absolute":
            self._post_actions = np.asarray(actions, dtype=np.float32)

    def record_velocity(self, *, stage: str, vel: np.ndarray, arm_mode: str) -> None:
        if not self.enabled or self.frame is None:
            return
        if stage == "raw_model_output":
            self._raw_velocities = np.asarray(vel, dtype=np.float32)
        elif stage == "postprocessed":
            self._post_velocities = np.asarray(vel, dtype=np.float32)

    def finalize_frame(self, output: Dict[str, Any], arm_mode: str) -> None:
        if not self.enabled or self.frame is None:
            return

        left_plot, right_plot = self._plot_arm_debug_panels(arm_mode=arm_mode)
        self.frame["left_arm_plot_path"] = left_plot
        self.frame["right_arm_plot_path"] = right_plot
        self.frame["output_action_shape"] = list(output["actions"].shape)

        step_json = os.path.join(self.output_dir, f"step_{self.step:06d}.json")
        with open(step_json, "w", encoding="utf-8") as f:
            json.dump(self.frame, f, ensure_ascii=False, indent=2)
        with open(self.latest_json, "w", encoding="utf-8") as f:
            json.dump(self.frame, f, ensure_ascii=False, indent=2)

        self.frame = None
        self._raw_actions = None
        self._post_actions = None
        self._raw_velocities = None
        self._post_velocities = None

    def _array_stats(self, arr: Any) -> Dict[str, Any]:
        if arr is None:
            return {"exists": False}
        if torch.is_tensor(arr):
            a = arr.detach().float().cpu().numpy()
            dtype = str(arr.dtype)
        else:
            a = np.asarray(arr)
            dtype = str(a.dtype)
        return {
            "exists": True,
            "dtype": dtype,
            "shape": list(a.shape),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
        }

    def _save_image(self, img: Any, filename: str, is_tensor: bool = False) -> str:
        if torch.is_tensor(img):
            arr = img.detach().float().cpu().numpy()
        else:
            arr = np.asarray(img)

        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            if arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)

        arr = arr.astype(np.float32)
        if is_tensor:
            arr_min, arr_max = float(arr.min()), float(arr.max())
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            if arr.max() > 1.5:
                arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        out_path = os.path.join(self.assets_dir, filename)
        plt.figure(figsize=(4, 4))
        plt.imshow(arr)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        return os.path.relpath(out_path, self.output_dir)

    def _split_actions_by_arm(self, arr: np.ndarray, arm_mode: str) -> Dict[str, Optional[np.ndarray]]:
        dim = arr.shape[-1]
        if dim == 14:
            return {"left": arr[:, :7], "right": arr[:, 7:14]}
        if dim == 7:
            if arm_mode == "right":
                return {"left": None, "right": arr}
            return {"left": arr, "right": None}
        return {"left": None, "right": None}

    def _get_arm_data(self, arr: Optional[np.ndarray], arm_name: str, arm_mode: str) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        split = self._split_actions_by_arm(arr, arm_mode=arm_mode)
        return split.get(arm_name)

    def _plot_arm_debug_panels(self, arm_mode: str) -> tuple[str, str]:
        return (
            self._plot_single_arm_panels("left", arm_mode=arm_mode),
            self._plot_single_arm_panels("right", arm_mode=arm_mode),
        )

    def _plot_single_arm_panels(self, arm_name: str, arm_mode: str) -> str:
        pos_raw = self._get_arm_data(self._raw_actions, arm_name=arm_name, arm_mode=arm_mode)
        pos_post = self._get_arm_data(self._post_actions, arm_name=arm_name, arm_mode=arm_mode)
        vel_raw = self._get_arm_data(self._raw_velocities, arm_name=arm_name, arm_mode=arm_mode)
        vel_post = self._get_arm_data(self._post_velocities, arm_name=arm_name, arm_mode=arm_mode)

        panel_defs: list[tuple[str, Optional[np.ndarray], str]] = [
            ("Position (Raw, Normalized)", pos_raw, "pos_raw"),
            ("Position (Postprocessed, Absolute)", pos_post, "pos_post"),
            ("Velocity (Raw)", vel_raw, "vel_raw"),
            ("Velocity (Postprocessed)", vel_post, "vel_post"),
        ]

        if all(data is None for _, data, _ in panel_defs):
            return ""

        joint_names = [f"j{i}" for i in range(6)] + ["gripper"]
        rows = 7
        cols = len(panel_defs)
        fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.1 * rows), sharex=True)

        for col_idx, (title, data, panel_type) in enumerate(panel_defs):
            for row_idx in range(rows):
                ax = axs[row_idx, col_idx]
                if data is not None and row_idx < data.shape[1]:
                    x = np.arange(data.shape[0])
                    ax.plot(x, data[:, row_idx], linewidth=1.0)
                elif data is None:
                    ax.set_axis_off()
                    continue

                if panel_type == "pos_raw":
                    ax.set_ylim(-1.0, 1.0)
                elif panel_type == "pos_post":
                    if row_idx == 6:
                        ax.set_ylim(-1.0, 1.0)
                    else:
                        ax.set_ylim(-float(np.pi), float(np.pi))

                if row_idx == 0:
                    ax.set_title(title)
                if col_idx == 0:
                    ax.set_ylabel(joint_names[row_idx])
                ax.grid(alpha=0.3)

        for col_idx in range(cols):
            axs[-1, col_idx].set_xlabel("t")

        fig.suptitle(f"{arm_name.capitalize()} Arm", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        filename = f"step{self.step:06d}_{arm_name}_arm_panels.png"
        out_path = os.path.join(self.assets_dir, filename)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return os.path.relpath(out_path, self.output_dir)

    def _write_dashboard_html(self) -> None:
        html_path = os.path.join(self.output_dir, "index.html")
        html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>NIAF VLA Debug Dashboard</title>
  <style>
    body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}
    h1,h2{margin:8px 0 12px}
    .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}
    .card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:10px}
    .meta{font-size:13px;line-height:1.5;white-space:pre-wrap}
    img{width:100%;border-radius:6px;border:1px solid #333;background:#000}
    .small{font-size:12px;color:#bbb}
  </style>
</head>
<body>
  <h1>NIAF VLA Debug Dashboard</h1>
  <div class=\"small\">Manual refresh (no auto reload).</div>
  <button id=\"refreshBtn\" style=\"margin:8px 0 12px;padding:6px 12px;cursor:pointer;\">Refresh</button>
  <h2>Summary</h2>
  <div id=\"summary\" class=\"card meta\"></div>
  <h2>Cameras (Raw / Preprocessed)</h2>
  <div id=\"cams\" class=\"row\"></div>
  <h2>Arm Curves</h2>
  <div class=\"row\">
    <div class=\"card\"><div class=\"small\">Left arm panels</div><img id=\"left_arm_plot\" /></div>
    <div class=\"card\"><div class=\"small\">Right arm panels</div><img id=\"right_arm_plot\" /></div>
  </div>

  <script>
  function formatSize(shape){
    if(!shape || !Array.isArray(shape) || shape.length < 2) return '-';
    const h = shape[shape.length - 2];
    const w = shape[shape.length - 1];
    if(typeof h === 'number' && typeof w === 'number') return `${h}x${w}`;
    return '-';
  }

  function fmtNum(v){
    if(v === undefined || v === null || Number.isNaN(v)) return '-';
    return Number(v).toFixed(4);
  }

  async function refresh(){
    try{
      const r = await fetch('latest.json?t=' + Date.now());
      if(!r.ok) return;
      const d = await r.json();

      const s = {
        step: d.step,
        timestamp: d.timestamp,
        model_name: d.model_name,
        action_mode: d.action_mode,
        arm_mode: d.arm_mode,
        chunk_size: d.chunk_size,
        target_chunk_size: d.target_chunk_size,
        active_cameras: d.active_cameras,
        raw_prompt: d.raw_prompt,
        formatted_prompt: d.formatted_prompt,
      };
      document.getElementById('summary').textContent = JSON.stringify(s, null, 2);

      const cams = d.cameras || [];
      const camsEl = document.getElementById('cams');
      camsEl.innerHTML = '';
      cams.forEach(c => {
        const div = document.createElement('div');
        div.className = 'card';
        const rawSize = formatSize(c.raw?.shape);
        const prepSize = formatSize(c.preprocessed?.shape);
        div.innerHTML = `
          <div><b>${c.camera_name}</b></div>
          <div class="small">raw: min=${fmtNum(c.raw?.min)}, max=${fmtNum(c.raw?.max)}, size=${rawSize}</div>
          <div class="small">preprocessed: min=${fmtNum(c.preprocessed?.min)}, max=${fmtNum(c.preprocessed?.max)}, size=${prepSize}</div>
          <div class="row" style="grid-template-columns:1fr 1fr;margin-top:8px">
            <div><div class="small">raw</div><img src="${c.raw_image_path}?t=${Date.now()}" /></div>
            <div><div class="small">preprocessed</div><img src="${c.preprocessed_image_path}?t=${Date.now()}" /></div>
          </div>
        `;
        camsEl.appendChild(div);
      });

      document.getElementById('left_arm_plot').src = (d.left_arm_plot_path || '') + '?t=' + Date.now();
      document.getElementById('right_arm_plot').src = (d.right_arm_plot_path || '') + '?t=' + Date.now();
    } catch(e) {}
  }

  document.getElementById('refreshBtn').addEventListener('click', refresh);
  refresh();
  </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
