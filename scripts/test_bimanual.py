#!/usr/bin/env python3
"""Test script for bimanual panda with PyBullet visualization."""

import vamp
from vamp import pybullet_interface as vpb
import time
import numpy as np

def sample_valid(robot_module, rng):
    """Sample a valid configuration."""
    for _ in range(1000):
        config = rng.next()
        if robot_module.validate(config):
            return config
    return None

def visualize_collision_spheres(sim, robot, config):
    """Draw collision spheres in PyBullet."""
    spheres = robot.fk(config)
    sphere_ids = []

    for i, sphere in enumerate(spheres):
        # Color spheres by arm (first half vs second half)
        if i < robot.n_spheres() // 2:
            color = [0.2, 0.6, 1.0, 0.3]  # Blue for arm 1
        else:
            color = [1.0, 0.4, 0.2, 0.3]  # Orange for arm 2

        sphere_id = sim.add_sphere(
            radius=sphere.r,
            position=[sphere.x, sphere.y, sphere.z],
            color=color
        )
        sphere_ids.append(sphere_id)

    return sphere_ids

def update_collision_spheres(sim, robot, config, sphere_ids):
    """Update positions of collision sphere visualizations."""
    spheres = robot.fk(config)
    for sphere_id, sphere in zip(sphere_ids, spheres):
        sim.update_object_position(sphere_id, [sphere.x, sphere.y, sphere.z])

def main(num_plans=5):
    robot = vamp.bimanualpanda70180
    print(f'Robot: bimanualpanda70180, {robot.dimension()} DOF, {robot.n_spheres()} spheres')
    print(f'End effectors: {robot.end_effectors()}')

    # Create PyBullet simulator
    print('\nInitializing PyBullet...')
    sim = vpb.PyBulletSimulator(
        'resources/bimanualpanda70180/bimanualpanda70180_spherized.urdf',
        robot.joint_names(),
        visualize=True
    )
    print('PyBullet ready!')

    # Setup planning
    rng = robot.halton()
    env = vamp.Environment()
    settings = vamp.AORRTCSettings()
    simp_settings = vamp.SimplifySettings()

    # Sample initial start
    print('\nSampling initial configuration...')
    #start = sample_valid(robot, rng)
    start = np.array([-1.8721524 , -0.43074113,  0.54297173, -1.6331258 , -0.5162828 ,
        2.7372644 ,  1.4881228 ,  0.10029678,  0.8015321 ,  1.3946488 ,
       -1.688786  , -0.51211363,  3.359184  ,  0.49604094], dtype=np.float32)
    
    start_segfault = np.array([-1.76001307,  0.65496548,  1.56808963, -1.72458956, -2.24366457,
        1.66725215,  2.48020473,  2.59006965, -0.77688433, -1.60382714,
       -2.39259848,  2.30752985,  2.56855721, -2.48020473], dtype=np.float32)
    if start is None:
        print('Failed to find valid start!')
        return

    # Set initial joint positions and create collision sphere visualizations
    sim.set_joint_positions(start.tolist())
    print('Creating collision sphere visualizations...')
    sphere_ids = visualize_collision_spheres(sim, robot, start)
    print(f'Created {len(sphere_ids)} collision spheres (blue=arm1, orange=arm2)')

    print(f'\nPlanning and visualizing {num_plans} motions...\n')

    for plan_num in range(num_plans):
        # Sample goal
        #goal = sample_valid(robot, rng)
        goal = np.array([ 0.33895347,  1.5497141 , -1.1819217 , -2.1733024 ,  1.0289133 ,
        3.580486  ,  1.4881228 ,  2.6405675 , -0.42822266, -1.181161  ,
       -1.470809  ,  0.23552905,  2.8322597 ,  0.16534698], dtype=np.float32)
        
        goal_segfault = np.array([-1.76350217, -0.2741152 ,  0.27893605, -1.59718334, -0.30652312,
        2.88686604,  1.15742887,  0.45545234,  1.46214723,  1.34557365,
       -1.27495863, -0.76661815,  3.3655516 ,  0.16534698], dtype=np.float32)

        if goal is None:
            print(f'Motion {plan_num+1}: Failed to find valid goal, skipping')
            continue
        assert robot.validate(start_segfault)
        assert robot.validate(goal_segfault)
        assert robot.validate(start)
        assert robot.validate(goal)
        
        # Plan
        result = robot.aorrtc(start_segfault, goal_segfault, env, settings, rng)

        if result.solved:
            # Simplify
            simplified = robot.simplify(result.path, env, simp_settings, rng)

            print(f'Motion {plan_num+1}/{num_plans}:')
            print(f'  Planning time: {result.nanoseconds/1000:.1f} μs')
            print(f'  Path length: {len(result.path)} -> {len(simplified.path)} waypoints')
            print(f'  Path cost: {result.path.cost():.3f} -> {simplified.path.cost():.3f}')

            # Interpolate for smooth visualization
            path = simplified.path
            path.interpolate_to_resolution(robot.resolution())

            print(f'  Playing {len(path)} interpolated waypoints...')
            # Animate with collision sphere updates
            for waypoint_idx in range(len(path)):
                config = path[waypoint_idx]
                if isinstance(config, np.ndarray):
                    config_list = config.tolist()
                else:
                    config_list = config.to_list()

                sim.set_joint_positions(config_list)
                update_collision_spheres(sim, robot, config, sphere_ids)
                time.sleep(0.016)  # ~60 fps

            # Set goal as new start
            # start = goal

            # Short pause between motions
            time.sleep(0.5)
        else:
            print(f'Motion {plan_num+1}: Failed to solve!')

    print(f'\n✓ Completed {num_plans} motions!')
    print('Keeping window open for 5 seconds...')
    time.sleep(5)

    # Clean exit
    print('Attempting to close PyBullet...')

if __name__ == '__main__':
    main(num_plans=5)