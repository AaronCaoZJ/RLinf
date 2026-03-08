#!/usr/bin/env python3
"""
MimicGen data generation script for BlockPAP-v1 (pick-and-place).

Video is generated AFTER generation by replaying saved states (like MimicGen
official), not during generation, to avoid slowdown.

Given --output /path/to/blockpap_gen.hdf5, the script automatically writes:
  /path/to/blockpap_gen.hdf5          successful demos
  /path/to/blockpap_gen_failed.hdf5   failed demos

Usage:
    cd /workspace1/zhijun/RLinf
    python real_franka/real2sim_env/mg_generate_blockpap_data.py \
        --src    .../blockpap_src.hdf5 \
        --output .../blockpap_gen.hdf5 \
        --num-demos 50 \
        --video-success .../success.mp4 \
        --video-failed  .../failed.mp4
"""
import argparse
import json
import os
import sys
import traceback

import h5py
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mg_blockpap_interface  # noqa: F401 – registers MG_BlockPAP
import pick_and_place          # noqa: F401 – registers BlockPAP-v1

from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface
import mimicgen.utils.file_utils as MG_FileUtils


def create_task_spec():
    """
    2-subtask task spec — mirrors MimicGen's official Stack task convention:

      1. Grasp (ref: block, term: grasped)
         offset=(10,20) naturally covers the lift motion after the grasp
         signal fires, so the arm is already elevated at the transition.

      2. Place (ref: target_coaster, term: None)
         5-step interpolation bridges from the lifted position to the
         start of the coaster-relative place trajectory.
    """
    spec = MG_TaskSpec()
    spec.add_subtask(
        object_ref="block",
        subtask_term_signal="grasped",
        # 10-20 extra frames after "grasped" fires — covers the lift motion,
        # matching MimicGen Stack's offset=(10,20) on the grasp subtask.
        subtask_term_offset_range=(10, 20),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=3),
        action_noise=1e-6,
        num_interpolation_steps=5,
        num_fixed_steps=0,
    )
    spec.add_subtask(
        object_ref="target_coaster",
        subtask_term_signal=None,
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=3),
        action_noise=1e-6,
        num_interpolation_steps=5,
        num_fixed_steps=0,
    )
    return spec


def _failed_path(output_path):
    """Derive the failed HDF5 path from the success HDF5 path.
    e.g. blockpap_gen.hdf5 → blockpap_gen_failed.hdf5
    """
    base, ext = os.path.splitext(output_path)
    return base + "_failed" + ext


def write_demos_to_hdf5(path, results, env):
    """Write a list of result dicts to an HDF5 file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["total"] = len(results)
        data_grp.attrs["env_args"] = json.dumps(env.serialize())

        for i, result in enumerate(results):
            ep_grp = data_grp.create_group(f"demo_{i}")
            ep_grp.create_dataset("actions", data=np.array(result["actions"]))
            # waypoint.py stores env.get_state()["states"] directly (raw array)
            ep_grp.create_dataset("states", data=np.array(result["states"]))

            ep_grp.attrs["num_samples"] = len(result["actions"])
            if result.get("src_demo_inds"):
                ep_grp.create_dataset("src_demo_inds",
                                      data=np.array(result["src_demo_inds"]))
            if "src_demo_labels" in result and len(result["src_demo_labels"]) > 0:
                ep_grp.create_dataset("src_demo_labels",
                                      data=np.array(result["src_demo_labels"]))

            obs_list = result.get("observations", [])
            if obs_list:
                obs_grp = ep_grp.create_group("obs")
                for key in obs_list[0].keys():
                    obs_grp.create_dataset(
                        key, data=np.array([o[key] for o in obs_list])
                    )

            # Write datagen_info — mirrors MimicGen's write_demo_to_hdf5 in file_utils.py
            datagen_info_list = result.get("datagen_infos", [])
            if datagen_info_list:
                dg = TensorUtils.list_of_flat_dict_to_dict_of_list(
                    [x.to_dict() for x in datagen_info_list]
                )
                for k in dg:
                    if k in ("object_poses", "subtask_term_signals"):
                        dg[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(dg[k])
                        for k2 in dg[k]:
                            ep_grp.create_dataset(
                                f"datagen_info/{k}/{k2}",
                                data=np.array(dg[k][k2]),
                            )
                    else:
                        ep_grp.create_dataset(
                            f"datagen_info/{k}",
                            data=np.array(dg[k]),
                        )
                ep_grp["datagen_info"].attrs["env_interface_name"] = "MG_BlockPAP"
                ep_grp["datagen_info"].attrs["env_interface_type"] = "maniskill"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",           type=str, required=True)
    parser.add_argument("--output",        type=str, required=True,
                        help="HDF5 path for successful demos; failed demos saved to "
                             "<stem>_failed.hdf5 in the same directory")
    parser.add_argument("--num-demos",     type=int, default=200)
    parser.add_argument("--max-attempts",  type=int, default=99999)
    parser.add_argument("--cam-t",         type=str, default="og")
    parser.add_argument("--use-image-obs", action="store_true", default=False,
                        help="Collect and save image observations in HDF5")
    parser.add_argument("--video-success", type=str, default=None,
                        help="Replay successful demos to this .mp4 (after generation)")
    parser.add_argument("--video-failed",  type=str, default=None,
                        help="Replay failed demos to this .mp4 (after generation)")
    parser.add_argument("--video-skip",    type=int, default=1,
                        help="Render every Nth frame during replay (default: 1)")
    parser.add_argument("--num-render",    type=int, default=None,
                        help="Max demos to include in each video (default: all)")
    args = parser.parse_args()

    # Derive failed HDF5 path automatically from --output
    output_failed = _failed_path(args.output)

    task_spec = create_task_spec()
    print(f"Task spec: {len(task_spec.spec)} subtasks")
    for i, s in enumerate(task_spec.spec):
        print(f"  Subtask {i}: object_ref={s['object_ref']}, "
              f"term_signal={s['subtask_term_signal']}")

    print(f"\nLoading source dataset: {args.src}")
    demo_keys = MG_FileUtils.get_all_demos_from_dataset(args.src)
    print(f"Source demos: {len(demo_keys)}")

    datagen_infos, _, _, _ = \
        MG_FileUtils.parse_source_dataset(
            dataset_path=args.src,
            demo_keys=demo_keys,
            task_spec=task_spec,
        )
    print(f"Parsed {len(datagen_infos)} source demos with subtask indices")

    pick_and_place.TRAJ_ID = "random"

    from mg_maniskill_wrapper import EnvManiskillBlockPAP
    env = EnvManiskillBlockPAP(
        env_name="BlockPAP-v1",
        render=False,
        render_offscreen=args.use_image_obs,
        use_image_obs=args.use_image_obs,
        cam_t=args.cam_t,
    )
    print(f"Created environment: {env.name}")

    env_interface = make_interface(
        name="MG_BlockPAP",
        interface_type="maniskill",
        env=env.base_env,
    )
    print(f"Created env interface: {env_interface}")

    # ── 建立 traj_id → source demo index 映射 ──────────────────────────────
    # 源 HDF5 按 --trajs 顺序写入 demo_0, demo_1, ...，每个 demo 的
    # real_traj_id 属性记录了对应的轨迹 ID（"0", "15", "25", "40", "45"）。
    traj_id_to_src_idx = {}
    with h5py.File(args.src, "r") as f_src:
        for i, dk in enumerate(sorted(f_src["data"].keys())):
            real_tid = f_src[f"data/{dk}"].attrs.get("real_traj_id", None)
            if real_tid is not None:
                traj_id_to_src_idx[str(real_tid)] = i
    print(f"Traj ID → source demo index mapping: {traj_id_to_src_idx}")

    data_gen = DataGenerator(
        task_spec=task_spec,
        dataset_path=args.src,
        demo_keys=demo_keys,
    )

    # ── Monkey-patch select_source_demo 以强制使用最近配置对应的源轨迹 ────
    _orig_select = data_gen.select_source_demo

    def _forced_select(eef_pose, object_pose, subtask_ind, src_subtask_inds,
                       subtask_object_name, selection_strategy_name,
                       selection_strategy_kwargs=None):
        """Override: always use the source demo matching the nearest traj config."""
        forced_tid = getattr(env.base_env, "_nearest_traj_id", None)
        if forced_tid is not None and forced_tid in traj_id_to_src_idx:
            return traj_id_to_src_idx[forced_tid]
        # Fallback to original selection if no mapping found
        return _orig_select(
            eef_pose=eef_pose, object_pose=object_pose,
            subtask_ind=subtask_ind, src_subtask_inds=src_subtask_inds,
            subtask_object_name=subtask_object_name,
            selection_strategy_name=selection_strategy_name,
            selection_strategy_kwargs=selection_strategy_kwargs,
        )

    data_gen.select_source_demo = _forced_select

    generated = []  # successful results
    failed    = []  # failed results
    num_attempts = 0

    print(f"\nGenerating {args.num_demos} demonstrations (no live rendering)...")
    while len(generated) < args.num_demos:
        num_attempts += 1

        try:
            # data_gen.generate() calls env.reset() internally, which sets
            # env.base_env._nearest_traj_id to the nearest config.
            # Our patched select_source_demo forces the matching source demo.
            result = data_gen.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=False,
                transform_first_robot_pose=True,
                interpolate_from_last_target_pose=True,
                video_writer=None,
                video_skip=args.video_skip,
                camera_names=None,
            )

            # result["initial_state"] is already set by data_gen.generate()

            if result["success"]:
                generated.append(result)
                nearest_tid = getattr(env.base_env, "_nearest_traj_id", "?")
                src_idx = traj_id_to_src_idx.get(nearest_tid, "?")
                print(f"  [{len(generated)}/{args.num_demos}] Success "
                      f"(attempt {num_attempts}, T={len(result['actions'])}, "
                      f"nearest_traj={nearest_tid}, src_demo={src_idx})")
            else:
                if len(result["states"]) > 0:
                    failed.append(result)
                if num_attempts % 10 == 0:
                    print(f"  Attempt {num_attempts}: failed "
                          f"({len(generated)}/{args.num_demos} so far)")

        except Exception as e:
            print(f"  Attempt {num_attempts}: error - {e}")
            traceback.print_exc()

        if num_attempts >= args.max_attempts:
            print(f"\nReached max attempts ({num_attempts}). "
                  f"Generated {len(generated)}/{args.num_demos} demos.")
            break

    # ── Write HDF5 files ──────────────────────────────────────────────────────
    print(f"\nWriting {len(generated)} successful demos to {args.output}")
    write_demos_to_hdf5(args.output, generated, env)

    print(f"Writing {len(failed)} failed demos to {output_failed}")
    write_demos_to_hdf5(output_failed, failed, env)

    print(f"Done! Generated {len(generated)} demos in {num_attempts} attempts.")
    print(f"Success rate: {len(generated)/num_attempts*100:.1f}%")

    # ── Post-generation replay videos ─────────────────────────────────────────
    # Both success and failed use the same replay_hdf5_to_video function.
    env.env.close()

    from mg_render_video import render_hdf5_to_video

    need_video = (args.video_success is not None and len(generated) > 0) or \
                 (args.video_failed is not None and len(failed) > 0)

    if need_video:
        print("\nRendering replay videos...")
        if args.video_success is not None and len(generated) > 0:
            render_hdf5_to_video(
                hdf5_path   = args.output,
                video_path  = args.video_success,
                cam_t       = args.cam_t,
                video_skip  = args.video_skip,
                num_renders = args.num_render,
            )

        if args.video_failed is not None and len(failed) > 0:
            render_hdf5_to_video(
                hdf5_path   = output_failed,
                video_path  = args.video_failed,
                cam_t       = args.cam_t,
                video_skip  = args.video_skip,
                num_renders = args.num_render,
            )


if __name__ == "__main__":
    main()
