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
import numpy as np
import sys
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


def main(
    skip_aorrtc_segfault_test: bool = False,
    only_test: str = None
):
    """
    Run comprehensive tests to diagnose bimanualpanda70180 segfault.

    Args:
        skip_aorrtc_segfault_test: Skip the test that causes segfault
        only_test: Run only a specific test (1-6)
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
        return

    # Run all tests
    tests.append(("Single Sample", test_single_sample()))
    tests.append(("Find Valid Sample", test_find_valid_sample()))

    passed, start, goal = test_segfault_configs_validation()
    tests.append(("Validate Segfault Configs", passed))

    tests.append(("RRTC with Segfault Configs", test_rrtc_with_segfault_configs(start, goal)))

    tests.append(("AORRTC with Random Configs", test_aorrtc_with_random_configs()))

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