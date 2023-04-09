import glfw
from OpenGL.GL import *
import numpy as np

gL = np.identity(3)  

def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # draw cooridnate
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()
    # draw triangle
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv( (T @ np.array([.0,.5,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.0,.0,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.5,.0,1.]))[:-1] )
    glEnd()


def key_callback(window, key, scancode, action, mods):
    global gL  
    if key==glfw.KEY_Q:
        if action==glfw.PRESS:
            newarray = np.array([[1., 0., -0.1],
                             [0., 1., 0.],
                             [0., 0., 1.]])
            gL = newarray @ gL  

    if key==glfw.KEY_E:
        if action==glfw.PRESS:
            newarray = np.array([[1., 0., 0.1],
                             [0., 1., 0.],
                             [0., 0., 1.]])
            gL = newarray @ gL 

    if key == glfw.KEY_A:
        if action == glfw.PRESS:

            theta = np.radians(10)
            c, s = np.cos(theta), np.sin(theta)
            newarray = np.array([[c, -s, 0.],
                            [s, c, 0.],
                            [0., 0., 1.]])
            gL = gL @ newarray


    if key == glfw.KEY_D:
        if action == glfw.PRESS:
            theta = np.radians(-10)
            c, s = np.cos(theta), np.sin(theta)
            newarray = np.array([[c, -s, 0.],
                            [s, c, 0.],
                            [0., 0., 1.]])
            gL = gL @ newarray

    if key==glfw.KEY_1:
        if action==glfw.PRESS:
            gL = np.identity(3)

    if key==glfw.KEY_W:
        if action==glfw.PRESS:
            newarray = np.array([[0.9, 0., 0.],
                             [0., 1., 0.],
                             [0., 0., 1.]])
            gL = gL @ newarray

    if key==glfw.KEY_S:
        if action==glfw.PRESS:
            theta = np.radians(10)
            c, s = np.cos(theta), np.sin(theta)
            newarray = np.array([[c, -s, 0.],
                            [s, c, 0.],
                            [0., 0., 1.]])
            gL = newarray @ gL


def main():
    if not glfw.init():
        return
    
    window = glfw.create_window(480, 480, "2021036835-3-1", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)
    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        render(gL)  # gL을 전달

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
