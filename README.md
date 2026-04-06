# isaac_table

Simple Isaac Sim tabletop bin-picking scene with an overhead RGB-D camera.

## Files

- `simple_floor_table.backup.py`: baseline scene with table, side bin, lighting, and camera.
- `simple_floor_table.py`: clutter drop pipeline using procedural objects, with RGB and depth capture after the objects settle in the bin.

## Run

From the Isaac Sim install root:

```powershell
.\python.bat standalone_examples\user\isaac_table\simple_floor_table.py
```

Useful options:

```powershell
.\python.bat standalone_examples\user\isaac_table\simple_floor_table.py --num-objects 8 --seed 7
.\python.bat standalone_examples\user\isaac_table\simple_floor_table.py --object-source prims
.\\python.bat standalone_examples\user\isaac_table\simple_floor_table.py --object-source ycb --asset-root "C:/Users/Warra/Downloads/Assets/Isaac/5.1"
```

Captured outputs are written to `camera_debug/` after the dropped objects settle.

## YCB Assets

The script supports a local Isaac asset pack and expects the asset root to be the folder containing the top-level `Isaac` directory.

Expected local asset root:

```text
C:\Users\Warra\Downloads\Assets\Isaac\5.1
```

Expected YCB path under that root:

```text
C:\Users\Warra\Downloads\Assets\Isaac\5.1\Isaac\Props\YCB\Axis_Aligned_Physics
```

Get the assets from the official Isaac Sim 5.1.0 download page:

- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html

For YCB, the relevant archive is the `Materials & Props` asset pack.

## Notes

- `prims` works without any external assets.
- `ycb` uses local USD assets from the Isaac asset pack and drops them into the bin from above.
- The overhead camera is configured as a RealSense-style RGB-D camera centered over the side-mounted bin.
- The script saves one RGB/depth capture after the clutter settles, or after a timeout if the objects keep jittering.
