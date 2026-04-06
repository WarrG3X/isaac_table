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
```

Captured outputs are written to `camera_debug/` after the dropped objects settle.

## Notes

- `ycb` is planned as a future object source, but the current setup uses procedural objects because the Isaac asset pack is not installed locally.
- The overhead camera is configured as a RealSense-style RGB-D camera centered over the side-mounted bin.
