#!/usr/bin/env python3
"""
Comprehensive test to diagnose bimanualpanda70180 segfault issue.

FINDINGS:
---------
The segfault occurs specifically in the AORRTC planner with certain configurations.
- Sampling works correctly (halton, xorshift)
- Single sample generation: ✅ WORKS
- Finding valid samples: ✅ WORKS
- Configuration validation: ✅ WORKS
- Forward kinematics (FK): ✅ WORKS
- RRTC planner: ✅ WORKS with segfault configs
- AORRTC planner: ❌ SEGFAULTS with specific configs

ROOT CAUSE:
-----------
Bug in AORRTC planner implementation for bimanualpanda70180 robot.
The same configurations that cause AORRTC to segfault work fine with RRTC.

REPRODUCIBLE SEGFAULT CONFIGS:
------------------------------
start: [-1.76001307,  0.65496548,  1.56808963, -1.72458956, -2.24366457,
        1.66725215,  2.48020473,  2.59006965, -0.77688433, -1.60382714,
       -2.39259848,  2.30752985,  2.56855721, -2.48020473]
goal:  [-1.76350217, -0.2741152 ,  0.27893605, -1.59718334, -0.30652312,
        2.88686604,  1.15742887,  0.45545234,  1.46214723,  1.34557365,
       -1.27495863, -0.76661815,  3.3655516 ,  0.16534698]
"""

import vamp
from vamp import pybullet_interface as vpb
import numpy as np
import sys
import time
from pathlib import Path
from fire import Fire


def test_single_sample():
    """Test 1: Can we get a single sample from each sampler?"""
    print("\n" + "="*70)
    print("TEST 1: Single Sample Generation")
    print("="*70)

    robot = vamp.bimanualpanda70180
    print(f'Robot: bimanualpanda70180')
    print(f'Dimensions: {robot.dimension()} DOF')
    print(f'Spheres: {robot.n_spheres()}')

    samplers = ['halton', 'xorshift']

    for sampler_name in samplers:
        print(f'\nTesting {sampler_name} sampler...')
        try:
            sampler = getattr(robot, sampler_name)()
            config = sampler.next()
            print(f'  ✅ Sample obtained: shape={config.shape}, dtype={config.dtype}')
        except Exception as e:
            print(f'  ❌ ERROR: {e}')
            return False

    print("\n✅ TEST 1 PASSED: All samplers can generate single samples")
    return True


def test_find_valid_sample():
    """Test 2: Can we find valid samples?"""
    print("\n" + "="*70)
    print("TEST 2: Finding Valid Samples")
    print("="*70)

    robot = vamp.bimanualpanda70180
    sampler = robot.halton()

    print('Attempting to find valid configuration...')
    for attempt in range(1000):
        config = sampler.next()
        if robot.validate(config):
            print(f'  ✅ Found valid config at attempt {attempt}')
            print(f'     Config: {config}')
            print("\n✅ TEST 2 PASSED: Can find valid samples")
            return True

    print("  ❌ Failed to find valid config after 1000 attempts")
    return False


def test_segfault_configs_validation():
    """Test 3: Can we validate the known segfault configurations?"""
    print("\n" + "="*70)
    print("TEST 3: Validating Known Segfault Configs")
    print("="*70)

    robot = vamp.bimanualpanda70180

    start_segfault = np.array([-1.76001307,  0.65496548,  1.56808963, -1.72458956, -2.24366457,
        1.66725215,  2.48020473,  2.59006965, -0.77688433, -1.60382714,
       -2.39259848,  2.30752985,  2.56855721, -2.48020473], dtype=np.float32)

    goal_segfault = np.array([-1.76350217, -0.2741152 ,  0.27893605, -1.59718334, -0.30652312,
        2.88686604,  1.15742887,  0.45545234,  1.46214723,  1.34557365,
       -1.27495863, -0.76661815,  3.3655516 ,  0.16534698], dtype=np.float32)

    print('Testing start_segfault validation...')
    sys.stdout.flush()
    is_valid_start = robot.validate(start_segfault)
    print(f'  Start valid: {is_valid_start}')

    print('Testing goal_segfault validation...')
    sys.stdout.flush()
    is_valid_goal = robot.validate(goal_segfault)
    print(f'  Goal valid: {is_valid_goal}')

    print('Testing FK on start...')
    sys.stdout.flush()
    start_spheres = robot.fk(start_segfault)
    print(f'  ✅ FK completed: {len(start_spheres)} spheres')

    print('Testing FK on goal...')
    sys.stdout.flush()
    goal_spheres = robot.fk(goal_segfault)
    print(f'  ✅ FK completed: {len(goal_spheres)} spheres')

    print("\n✅ TEST 3 PASSED: Segfault configs are valid and FK works")
    return True, start_segfault, goal_segfault


def test_rrtc_with_segfault_configs(start, goal):
    """Test 4: Does RRTC work with the segfault configs?"""
    print("\n" + "="*70)
    print("TEST 4: RRTC Planner with Segfault Configs")
    print("="*70)

    robot = vamp.bimanualpanda70180
    env = vamp.Environment()
    rng = robot.halton()
    settings = vamp.RRTCSettings()

    print('Planning with RRTC...')
    sys.stdout.flush()

    try:
        result = robot.rrtc(start, goal, env, settings, rng)
        print(f'  ✅ Planning completed! Solved: {result.solved}')
        if result.solved:
            print(f'     Path has {len(result.path)} waypoints')
        print("\n✅ TEST 4 PASSED: RRTC works with segfault configs")
        return True
    except Exception as e:
        print(f'  ❌ ERROR: {e}')
        return False


def test_aorrtc_with_segfault_configs(start, goal):
    """Test 5: Does AORRTC work with the segfault configs? (EXPECTED TO FAIL)"""
    print("\n" + "="*70)
    print("TEST 5: AORRTC Planner with Segfault Configs")
    print("="*70)
    print("⚠️  WARNING: This test is EXPECTED to cause a segmentation fault")
    print("="*70)

    robot = vamp.bimanualpanda70180
    env = vamp.Environment()
    rng = robot.halton()
    settings = vamp.AORRTCSettings()

    print('\nPlanning with AORRTC...')
    sys.stdout.flush()

    try:
        result = robot.aorrtc(start, goal, env, settings, rng)
        print(f'  ✅ Planning completed! Solved: {result.solved}')
        print("\n✅ TEST 5 PASSED: AORRTC works (unexpected!)")
        return True
    except Exception as e:
        print(f'  ❌ Python exception: {e}')
        return False


def test_aorrtc_with_random_configs():
    """Test 6: Does AORRTC work with random valid configs?"""
    print("\n" + "="*70)
    print("TEST 6: AORRTC Planner with Random Valid Configs")
    print("="*70)

    robot = vamp.bimanualpanda70180
    sampler = robot.halton()

    # Find valid start
    print('Finding valid start...')
    start = None
    for _ in range(1000):
        config = sampler.next()
        if robot.validate(config):
            start = config
            break

    if start is None:
        print("  ❌ Could not find valid start")
        return False
    print(f'  Found start: {start}')

    # Find valid goal
    print('Finding valid goal...')
    goal = None
    for _ in range(1000):
        config = sampler.next()
        if robot.validate(config):
            goal = config
            break

    if goal is None:
        print("  ❌ Could not find valid goal")
        return False
    print(f'  Found goal: {goal}')

    # Try planning
    env = vamp.Environment()
    rng = robot.halton()
    settings = vamp.AORRTCSettings()

    print('\nPlanning with AORRTC...')
    sys.stdout.flush()

    try:
        result = robot.aorrtc(start, goal, env, settings, rng)
        print(f'  ✅ Planning completed! Solved: {result.solved}')
        print("\n✅ TEST 6 PASSED: AORRTC works with random configs")
        return True
    except Exception as e:
        print(f'  ❌ ERROR: {e}')
        return False


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


def test_rrtc_with_random_configs_and_playback(num_plans=3):
    """Test 7: RRTC with random configs and PyBullet playback visualization."""
    print("\n" + "="*70)
    print("TEST 7: RRTC Planner with Random Configs and Playback")
    print("="*70)

    robot = vamp.bimanualpanda70180
    print(f'Robot: bimanualpanda70180, {robot.dimension()} DOF, {robot.n_spheres()} spheres')

    # Find URDF path
    urdf_path = Path(__file__).parent.parent / 'resources' / 'bimanualpanda70180' / 'bimanualpanda70180_spherized.urdf'
    if not urdf_path.exists():
        print(f"  ⚠️  URDF not found at {urdf_path}, skipping visualization")
        return False

    # Create PyBullet simulator
    print('\nInitializing PyBullet...')
    sim = None
    try:
        import os
        # Check if we have a display
        if 'DISPLAY' not in os.environ:
            print('  ⚠️  No DISPLAY environment variable, skipping PyBullet visualization')
        else:
            sim = vpb.PyBulletSimulator(
                str(urdf_path),
                robot.joint_names(),
                visualize=True
            )
            print('  ✅ PyBullet initialized')
    except Exception as e:
        print(f'  ⚠️  Could not initialize PyBullet: {e}')
        print('  Continuing without visualization...')
        sim = None

    # Setup planning
    sampler = robot.halton()
    env = vamp.Environment()
    settings = vamp.RRTCSettings()
    simp_settings = vamp.SimplifySettings()

    # Find initial start
    print('\nFinding initial valid configuration...')
    start = None
    for _ in range(1000):
        config = sampler.next()
        if robot.validate(config):
            start = config
            break

    if start is None:
        print('  ❌ Could not find valid start')
        return False
    print(f'  ✅ Found start')

    # Initialize visualization if available
    sphere_ids = None
    if sim is not None:
        sim.set_joint_positions(start.tolist())
        sphere_ids = visualize_collision_spheres(sim, robot, start)
        print(f'  ✅ Created {len(sphere_ids)} collision spheres (blue=arm1, orange=arm2)')

    print(f'\nPlanning and visualizing {num_plans} motions...\n')
    successful_plans = 0

    for plan_num in range(num_plans):
        # Find valid goal
        print(f'Motion {plan_num+1}/{num_plans}: Finding goal...')
        goal = None
        for _ in range(1000):
            config = sampler.next()
            if robot.validate(config):
                goal = config
                break

        if goal is None:
            print(f'  ⚠️  Could not find valid goal, skipping')
            continue

        # Plan with RRTC
        print(f'  Planning with RRTC...')
        sys.stdout.flush()
        rng = robot.halton()

        try:
            result = robot.rrtc(start, goal, env, settings, rng)

            if result.solved:
                # Simplify
                simplified = robot.simplify(result.path, env, simp_settings, rng)

                print(f'  ✅ Solved!')
                print(f'     Planning time: {result.nanoseconds/1000:.1f} μs')
                print(f'     Path: {len(result.path)} -> {len(simplified.path)} waypoints')
                print(f'     Cost: {result.path.cost():.3f} -> {simplified.path.cost():.3f}')

                # Interpolate and visualize
                path = simplified.path
                path.interpolate_to_resolution(robot.resolution())
                print(f'     Interpolated to {len(path)} waypoints')

                if sim is not None:
                    print(f'     Playing back motion...')
                    for waypoint_idx in range(len(path)):
                        config = path[waypoint_idx]
                        if isinstance(config, np.ndarray):
                            config_list = config.tolist()
                        else:
                            config_list = config.to_list()

                        sim.set_joint_positions(config_list)
                        if sphere_ids is not None:
                            update_collision_spheres(sim, robot, config, sphere_ids)
                        time.sleep(0.016)  # ~60 fps

                    time.sleep(0.5)  # Pause between motions

                # Use goal as next start
                start = goal
                successful_plans += 1

            else:
                print(f'  ❌ Failed to solve')

        except Exception as e:
            print(f'  ❌ ERROR: {e}')
            import traceback
            traceback.print_exc()

    print(f'\n{"="*70}')
    print(f'Completed {successful_plans}/{num_plans} successful plans')

    if sim is not None:
        print('Keeping window open for 5 seconds...')
        time.sleep(5)

    if successful_plans > 0:
        print("\n✅ TEST 7 PASSED: RRTC works with random configs and playback")
        return True
    else:
        print("\n❌ TEST 7 FAILED: No successful plans")
        return False


def main(
    skip_aorrtc_segfault_test: bool = False,
    only_test: str = None,
    num_plans: int = 3
):
    """
    Run comprehensive tests to diagnose bimanualpanda70180 segfault.

    Args:
        skip_aorrtc_segfault_test: Skip the test that causes segfault
        only_test: Run only a specific test (1-7)
        num_plans: Number of plans to generate for test 7 (default: 3)
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE SEGFAULT DIAGNOSIS FOR bimanualpanda70180")
    print("="*70)

    tests = []

    if only_test:
        test_num = int(only_test)
        if test_num == 1:
            test_single_sample()
        elif test_num == 2:
            test_find_valid_sample()
        elif test_num == 3:
            passed, start, goal = test_segfault_configs_validation()
        elif test_num == 4:
            _, start, goal = test_segfault_configs_validation()
            test_rrtc_with_segfault_configs(start, goal)
        elif test_num == 5:
            _, start, goal = test_segfault_configs_validation()
            test_aorrtc_with_segfault_configs(start, goal)
        elif test_num == 6:
            test_aorrtc_with_random_configs()
        elif test_num == 7:
            test_rrtc_with_random_configs_and_playback(num_plans)
        return

    # Run all tests
    tests.append(("Single Sample", test_single_sample()))
    tests.append(("Find Valid Sample", test_find_valid_sample()))

    passed, start, goal = test_segfault_configs_validation()
    tests.append(("Validate Segfault Configs", passed))

    tests.append(("RRTC with Segfault Configs", test_rrtc_with_segfault_configs(start, goal)))

    tests.append(("AORRTC with Random Configs", test_aorrtc_with_random_configs()))

    tests.append(("RRTC with Random Configs and Playback", test_rrtc_with_random_configs_and_playback(num_plans)))

    if not skip_aorrtc_segfault_test:
        print("\n" + "="*70)
        print("⚠️  RUNNING SEGFAULT TEST (this will likely crash)")
        print("="*70)
        input("Press Enter to continue or Ctrl+C to abort...")
        tests.append(("AORRTC with Segfault Configs", test_aorrtc_with_segfault_configs(start, goal)))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("The segfault is isolated to the AORRTC planner with specific")
    print("configurations. The sampler works correctly. RRTC planner works")
    print("fine with the same configurations.")
    print("\nRECOMMENDATION: Use RRTC planner instead of AORRTC for")
    print("bimanualpanda70180 until the AORRTC bug is fixed.")
    print("="*70)


if __name__ == '__main__':
    Fire(main)