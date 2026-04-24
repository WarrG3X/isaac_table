# isaac_table

Isaac Sim tabletop data-collection scenes for Piper X, with the current focus on a targeted clutter-pick workflow.

![Rendered outputs](media/render_triptych.png)

![Full scene example](media/full_scene_example.PNG)

## Current Focus

The main workflow in this repo is now:

1. run the targeted clutter-pick Piper scene
2. teleoperate and record a raw episode
3. optionally replay the episode in Isaac Sim
4. export the raw episode to MCAP
5. inspect the result in Foxglove

The primary entrypoint is:

- [piper_x_clutter_pick_target_ps5.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\clutter_pick\piper_x_clutter_pick_target_ps5.py)

The older simple-pick scene still exists, but it is no longer the main data-collection path.

## Prerequisites

Tested setup:

- Isaac Sim `5.1.x`
- Windows

External dependencies expected by the current scripts:

- Isaac asset root:
  - `C:\Users\Warra\Downloads\Assets\Isaac\5.1`
- AgileX Piper repo cloned into the Isaac root:
  - `C:\Users\Warra\Downloads\Isaac\piper_isaac_sim`

Expected Piper USD used by the scenes:

```text
C:\Users\Warra\Downloads\Isaac\piper_isaac_sim\USD\piper_x_v1.usd
```

Why the shipped USD is used:

- on Windows, the cloned Piper repo has mesh filename case collisions such as `base_Link.dae` vs `base_link.dae`
- so the scenes use the shipped `piper_x_v1.usd` instead of re-importing the URDF

## Repo Layout

- [piper/clutter_pick](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\clutter_pick): main clutter-pick scenes, teleop, and debug tools
- [scenes](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\scenes): shared scene builders used by playback and scene variants
- [utils](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\utils): dataset/export/viewer utilities
- [playback_episode.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\playback_episode.py): replay a saved episode using the scene registry
- [debug](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\debug): robot/controller debug scripts
- [simple_floor_table.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\simple_floor_table.py): older standalone tabletop clutter scene

## Main Scene

The main targeted clutter-pick teleop scene is:

- [piper_x_clutter_pick_target_ps5.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\clutter_pick\piper_x_clutter_pick_target_ps5.py)

Current behavior:

- `Simple_Room` environment for more realistic room lighting
- Piper X teleop through PS5
- source and target trays
- targeted object setup for clutter-pick collection
- fixed wrist camera
- fixed world camera for the static/top view
- raw episode recording with scene metadata

Run it from the Isaac root:

```powershell
cd C:\Users\Warra\Downloads\Isaac
.\python.bat standalone_examples\user\isaac_table\piper\clutter_pick\piper_x_clutter_pick_target_ps5.py
```

Optional camera panels:

```powershell
.\python.bat standalone_examples\user\isaac_table\piper\clutter_pick\piper_x_clutter_pick_target_ps5.py --camera-panels
```

Useful current defaults in that script:

- target object: `061_foam_brick.usd`
- default object count: `12`
- shallower tray height for easier corner access: `0.05`

## PS5 Controls

In the targeted clutter-pick PS5 scene:

- `Cross` toggles the gripper
- `Circle` resets and re-randomizes the clutter arrangement
- `Square` starts/stops recording

The status window shows:

- current target pose
- save status
- live normalized gripper value

The normalized gripper value is also recorded into raw episodes as `gripper_scalar`.

## Raw Episode Output

Targeted clutter-pick episodes are written under:

```text
C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\clutter_pick_target_raw
```

Each episode directory contains:

- `data.npz`
- `metadata.json`
- `scene.yaml`
- `images/top/*`
- `images/wrist/*`

Important fields currently saved into `data.npz`:

- `timestamp`
- `joint_position`
- `ee_position`
- `ee_orientation_wxyz`
- `ee_delta`
- `gripper_action`
- `gripper_open`
- `gripper_scalar`
- `object_position`
- `object_orientation_wxyz`

## Playback

Saved episodes can be replayed through the shared scene registry:

```powershell
cd C:\Users\Warra\Downloads\Isaac
.\python.bat standalone_examples\user\isaac_table\playback_episode.py --episode-dir standalone_examples\user\isaac_table\data\clutter_pick_target_raw\episode_YYYYMMDD_HHMMSS
```

Useful options:

```powershell
.\python.bat standalone_examples\user\isaac_table\playback_episode.py --episode-dir ... --camera-panels
.\python.bat standalone_examples\user\isaac_table\playback_episode.py --episode-dir ... --auto-start
.\python.bat standalone_examples\user\isaac_table\playback_episode.py --episode-dir ... --mode full_state
```

Playback uses:

- [registry.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\scenes\registry.py)
- `scene.yaml` saved with each episode

## Dataset and Viewer Utilities

### Raw -> LeRobot

```powershell
cd C:\Users\Warra\Downloads\Isaac
python standalone_examples\user\isaac_table\utils\convert_to_lerobot.py --raw-dir C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\clutter_pick_target_raw\episode_YYYYMMDD_HHMMSS --output local/piper_clutter_pick --root C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\lerobot
```

### LeRobot -> Rerun

```powershell
cd C:\Users\Warra\Downloads\Isaac
python standalone_examples\user\isaac_table\utils\view_in_rerun.py --dataset local/piper_clutter_pick --root C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\lerobot --episode 0 --raw-dir C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\clutter_pick_target_raw\episode_YYYYMMDD_HHMMSS --save C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\lerobot\local\piper_clutter_pick\episode_0.rrd
```

### Raw -> MCAP

```powershell
cd C:\Users\Warra\Downloads\Isaac
.\python.bat standalone_examples\user\isaac_table\utils\convert_raw_to_mcap.py --episode-dir C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\data\clutter_pick_target_raw\episode_YYYYMMDD_HHMMSS
```

This writes:

- `<episode_dir>\<episode_name>.mcap`

The exporter lives here:

- [convert_raw_to_mcap.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\utils\convert_raw_to_mcap.py)

### Foxglove Layout

Reusable Foxglove layout for targeted clutter-pick MCAP episodes:

- [clutter_pick_target_layout.json](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\utils\foxglove\clutter_pick_target_layout.json)

It is set up for:

- top camera image
- wrist camera image
- EE position plot
- joint plots
- gripper plot
- raw object/joint/pose/metadata panels

## Debug / Scene Tuning

Useful current clutter-pick debug tools:

- [clutter_pick_target_debug.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\clutter_pick\clutter_pick_target_debug.py)
  - clutter-only target-scene reset/debug loop
- [temp_view_camera_probe.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\clutter_pick\temp_view_camera_probe.py)
  - probe and save the static camera sensor output while tuning the world-camera pose
- [piper_x_clutter_pick_ps5.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\clutter_pick\piper_x_clutter_pick_ps5.py)
  - non-targeted clutter-pick PS5 scene
- [piper_x_simple_pick_ps5.py](C:\Users\Warra\Downloads\Isaac\standalone_examples\user\isaac_table\piper\simple_pick\piper_x_simple_pick_ps5.py)
  - older simple pick/place teleop scene

Useful robot/controller debug scripts:

```powershell
.\python.bat standalone_examples\user\isaac_table\debug\piper_x_joint_debug.py
.\python.bat standalone_examples\user\isaac_table\debug\ps5_debug.py
.\python.bat standalone_examples\user\isaac_table\debug\ps5_scene_debug.py
.\python.bat standalone_examples\user\isaac_table\piper\bin_pick\piper_x_scene.py --ik-ui
```

## Asset Notes

Local YCB asset bank expected at:

```text
C:\Users\Warra\Downloads\Assets\Isaac\5.1\Isaac\Props\YCB
```

Important distinction on this machine:

- `Axis_Aligned_Physics`
  - small curated physics-ready subset
- `Axis_Aligned`
  - broader object bank
  - used with auto-generated convex colliders when broader clutter variety is needed

## Notes

- the Piper and Isaac asset dependencies are not self-contained in this repo
- camera panels are optional; recording still captures both camera feeds when panels are off
- raw-first recording is the intended workflow; LeRobot, Rerun, and MCAP are post-process exports
- the scene-builder/playback path is scene-agnostic, but targeted clutter-pick is the current main workflow
