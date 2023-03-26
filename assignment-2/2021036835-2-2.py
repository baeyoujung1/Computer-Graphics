import glfw
from OpenGL.GL import *
import numpy as np

p_type=GL_LINE_LOOP

def render():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glBegin(p_type)
    angles = np.linspace(0, 360, 12, False)
    for angle in angles:
        angle_radian = np.radians(angle)
        x = np.cos(angle_radian)
        y = np.sin(angle_radian)
        glVertex2f(x, y)
    glEnd()

def key_callback(window, key, scancode, action, mods):

    global p_type
    if key==glfw.KEY_1:
        if action==glfw.PRESS:
            p_type=GL_POINTS
    elif key==glfw.KEY_2:
        if action==glfw.PRESS:
            p_type=GL_LINES
    elif key==glfw.KEY_3:
        if action==glfw.PRESS:
            p_type=GL_LINE_STRIP
    elif key==glfw.KEY_4:
        if action==glfw.PRESS:
            p_type=GL_LINE_LOOP
    elif key==glfw.KEY_5:
        if action==glfw.PRESS:
            p_type=GL_TRIANGLES
    elif key==glfw.KEY_6:
        if action==glfw.PRESS:
            p_type=GL_TRIANGLE_STRIP
    elif key==glfw.KEY_7:
        if action==glfw.PRESS:
            p_type=GL_TRIANGLE_FAN
    elif key==glfw.KEY_8:
        if action==glfw.PRESS:
            p_type=GL_QUADS
    elif key==glfw.KEY_9:
        if action==glfw.PRESS:
            p_type=GL_QUAD_STRIP
    elif key==glfw.KEY_0:
        if action==glfw.PRESS:
            p_type=GL_POLYGON

def main():
    # Initialize the library
    if not glfw.init():
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(480,480,"2021036835-2-2", None,None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)

    # Make the window's context current
    glfw.make_context_current(window)

     # Set the background color to be the same as the clear color

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # Render here, e.g. using pyOpenGL
        render()

        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()

