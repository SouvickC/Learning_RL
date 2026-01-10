import mujoco
import mujoco.viewer
import numpy as np
import time
import mujoco_viewer


MODEL_PATH = r"D:\CS\RL\mujoco_menagerie\boston_dynamics_spot\scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# # create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen')

viewer.add_line_to_fig(line_name="root-pos-x", fig_idx=0)
viewer.add_line_to_fig(line_name="root-pos-z", fig_idx=0)
viewer.add_line_to_fig(line_name="right_ankle_y", fig_idx=1)

# user has access to mjvFigure
fig = viewer.figs[0]
fig.title = "Root Position"
fig.flg_legend = True
fig.xlabel = "Timesteps"
fig.figurergba[0] = 0.2
fig.figurergba[3] = 0.2
fig.gridsize[0] = 5
fig.gridsize[1] = 5

fig = viewer.figs[1]
fig.title = "Joint position"
fig.flg_legend = True
fig.figurergba[0] = 0.2
fig.figurergba[3] = 0.2


import numpy as np

def get_trot_gait(t, period=1.0, hip_range=0.6, knee_range=0.4, hip_offset=0.8, knee_offset=-1.5):
    """
    Generate target joint angles for trotting.
    t: current time (seconds)
    """
    # Normalize phase
    phase = (t % period) / period * 2 * np.pi
    
    # Swing = forward/up, Stance = backward/down
    fl_hr_phase = phase          # Front-left & hind-right
    fr_hl_phase = phase + np.pi  # Front-right & hind-left (opposite)

    # Hip flexion (hy): positive = forward
    fl_hy = hip_offset + hip_range * np.sin(fl_hr_phase)
    fr_hy = hip_offset + hip_range * np.sin(fr_hl_phase)
    hl_hy = hip_offset + hip_range * np.sin(fr_hl_phase)
    hr_hy = hip_offset + hip_range * np.sin(fl_hr_phase)

    # Knee (kn): more negative = more bent (lift leg during swing)
    fl_kn = knee_offset - knee_range * np.sin(fl_hr_phase)
    fr_kn = knee_offset - knee_range * np.sin(fr_hl_phase)
    hl_kn = knee_offset - knee_range * np.sin(fr_hl_phase)
    hr_kn = knee_offset - knee_range * np.sin(fl_hr_phase)

    # Keep hip abduction (hx) near 0 (legs under body)
    hx_vals = [0.0] * 4

    return np.array([
        hx_vals[0], fl_hy, fl_kn,
        hx_vals[1], fr_hy, fr_kn,
        hx_vals[2], hl_hy, hl_kn,
        hx_vals[3], hr_hy, hr_kn
    ])

start_time = time.time()
while viewer.is_alive:
    viewer.add_data_to_line(line_name="root-pos-x",
                            line_data=data.qpos[0], fig_idx=0)
    viewer.add_data_to_line(line_name="root-pos-z",
                            line_data=data.qpos[2], fig_idx=0)
    viewer.add_data_to_line(line_name="right_ankle_y", line_data=data.qpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_y")], fig_idx=1)
    # sim step
    t = time.time() - start_time
    data.ctrl[:] = get_trot_gait(t, period=0.8)
    mujoco.mj_step(model, data)

    time.sleep(0.01)
    # render
    viewer.render()
    if not viewer.is_alive:
        break



# # simulate and render
# for _ in range(100000):
#     viewer.add_data_to_line(line_name="root-pos-x",
#                             line_data=data.qpos[0], fig_idx=0)
#     viewer.add_data_to_line(line_name="root-pos-z",
#                             line_data=data.qpos[2], fig_idx=0)
#     viewer.add_data_to_line(line_name="right_ankle_y", line_data=data.qpos[
#         mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_y")], fig_idx=1)

#     mujoco.mj_step(model, data)
#     viewer.render()
#     if not viewer.is_alive:
#         break


# close
viewer.close()