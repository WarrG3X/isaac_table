from . import clutter_pick, simple_pick


SCENE_BUILDERS = {
    "simple_pick": simple_pick.build_scene,
    "clutter_pick": clutter_pick.build_scene,
}


def get_scene_builder(scene_name):
    if scene_name not in SCENE_BUILDERS:
        raise KeyError(f"Unknown scene '{scene_name}'. Available: {', '.join(sorted(SCENE_BUILDERS))}")
    return SCENE_BUILDERS[scene_name]


def list_scenes():
    return sorted(SCENE_BUILDERS)
