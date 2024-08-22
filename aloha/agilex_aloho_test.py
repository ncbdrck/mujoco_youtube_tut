import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = 'meshes_mujoco/aloha_v1.xml'  # XML file (assumes this is in the same folder as this file)
simend = 15  # Simulation time
print_camera_config = 0  # Set to 1 to print camera config, useful for initializing view of the model

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# EE pose for the left robot arm
def get_ee_pose_fl(data):
    """
    Get the end-effector pose for the left robot arm.
    Assuming that 'fl_link8' is the end-effector body name

    :param data: The MuJoCo data.
    :return:
    """

    ee_position = data.body('fl_link8').xpos
    ee_orientation = data.body('fl_link8').xmat
    return ee_position, ee_orientation

# EE pose for the right robot arm
def get_ee_pose_fr(data):
    """
    Get the end-effector pose for the right robot arm.
    Assuming that 'fr_link8' is the end-effector body name

    :param data:
    :return:
    """
    ee_position = data.body('fr_link8').xpos
    ee_orientation = data.body('fr_link8').xmat
    return ee_position, ee_orientation

# set the target positions for the robot joints for the left arm
def set_joint_positions_left_arm(model, data, target_positions):
    """
    Set the target positions for the robot joints for the left arm.
    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param target_positions: List or array of target joint positions.
    """
    joint_names = ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4',
                   'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8']

    for i, joint_name in enumerate(joint_names):
        if i < len(target_positions):
            actuator_id = model.actuator(joint_name).id
            data.ctrl[actuator_id] = target_positions[i]

# set the target positions for the robot joints for the right arm
def set_joint_positions_right_arm(model, data, target_positions):
    """
    Set the target positions for the robot joints for the right arm.
    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param target_positions: List or array of target joint positions.
    """
    joint_names = ['fr_joint1', 'fr_joint2', 'fr_joint3', 'fr_joint4',
                   'fr_joint5', 'fr_joint6', 'fr_joint7', 'fr_joint8']

    for i, joint_name in enumerate(joint_names):
        if i < len(target_positions):
            actuator_id = model.actuator(joint_name).id
            data.ctrl[actuator_id] = target_positions[i]


def visualize_camera(model, data, camera_name='f_dabai'):
    """
    Render the camera view specified by `camera_name`.
    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param camera_name: The name of the camera to visualize.

    We have the following cameras in the model:
    - f_dabai
    - fl_dabai
    - fr_dabai

    """
    # Create a new camera object for rendering
    cam = mj.MjvCamera()

    # Get the ID of the specified camera in the model
    camera_id = model.camera(camera_name).id

    # Set the camera settings based on the model's camera
    cam.fixedcamid = camera_id
    cam.type = mj.mjtCamera.mjCAMERA_FIXED

    # Create a new viewport for the camera view
    viewport_width, viewport_height = 320, 240
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update the scene for the camera view
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)

    # Render the scene from the camera's perspective
    mj.mjr_render(viewport, scene, context)

# similar to ros's joint_states topic
def get_joint_states(model, data, joint_names):
    """
    Retrieve the joint angles, velocities, accelerations, and torques for the specified joints.

    :param model: The MuJoCo model.
    :param data: The MuJoCo data.
    :param joint_names: List of joint names to retrieve states for.
    :return: A dictionary with joint states (angles, velocities, accelerations, and torques).
    """
    joint_states = {
        'angles': [],
        'velocities': [],
        'accelerations': [],
        'torques': []
    }

    for joint_name in joint_names:
        joint_id = model.joint(joint_name).qposadr  # Index in qpos array for this joint
        dof_id = model.joint(joint_name).dofadr  # Index in dof array for this joint

        # Joint angle (position)
        angle = data.qpos[joint_id]

        # Joint velocity
        velocity = data.qvel[dof_id]

        # Joint acceleration (MuJoCo doesn't directly provide accelerations, so we approximate it)
        acceleration = data.qacc[dof_id]

        # Joint torque
        torque = data.qfrc_actuator[dof_id]  # or use data.qfrc_applied[dof_id] for applied forces

        # Store the retrieved states
        joint_states['angles'].append(angle)
        joint_states['velocities'].append(velocity)
        joint_states['accelerations'].append(acceleration)
        joint_states['torques'].append(torque)

    return joint_states

# let's test the joint_states function
def test_joint_states(model, data):
    """
    Test the get_joint_states function by printing the joint states at each time step.
    """
    joint_names = ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4',
                   'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8']

    # Retrieve joint states
    joint_states = get_joint_states(model, data, joint_names)

    # Print joint states for testing
    print("Joint Angles:", joint_states['angles'])
    print("Joint Velocities:", joint_states['velocities'])
    print("Joint Accelerations:", joint_states['accelerations'])
    print("Joint Torques:", joint_states['torques'])
    print("\n")


def test_all_functions(model, data):
    # Get end-effector pose
    ee_pos_fl, ee_ori_fl = get_ee_pose_fl(data)
    print("End-Effector Position:", ee_pos_fl)
    print("End-Effector Orientation:", ee_ori_fl)
    test_joint_states(model, data)

    # Send commands to move the robot
    target_positions = [0.0, 0.5, 0.1, -0.2, 0.3, 2.0, 1.01, 0.01]
    set_joint_positions_left_arm(model, data, target_positions)
    test_joint_states(model, data)

def init_controller(model, data):
    # initialize the controller here. This function is called once, in the beginning
    pass


def controller(model, data):
    # put the controller here. This function is called inside the simulation.
    pass


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # Visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])

# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 1.0 / 60.0):
        mj.mj_step(model, data)
        test_all_functions(model, data)

    if data.time >= simend:
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # print camera configuration (help to initialize the view)
    if print_camera_config == 1:
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    # Update scene and render main view
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # Visualize camera view
    visualize_camera(model, data, camera_name='f_dabai')
    visualize_camera(model, data, camera_name='fl_dabai')
    visualize_camera(model, data, camera_name='fr_dabai')

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
