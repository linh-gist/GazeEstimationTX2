import numpy as np
import cv2
import threading
import datetime
import pickle

from prompt_toolkit.keys import Keys
import gi.repository
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk

class cell:
    def __init__(self):
        self.width = 170 # mm
        self.height = 230  # mm
        self.center = (0, 0)
        self.color = (0, 0, 255) # red
        self.thickness = 2
    def draw(self, image, ratio = 1):
        start_point = (int((self.center[0]-self.width/2)*ratio), int((self.center[1]-self.height/2)*ratio))
        end_point = (int((self.center[0]+self.width/2)*ratio), int((self.center[1]+self.height/2)*ratio))
        image = cv2.rectangle(image, start_point, end_point, self.color, self.thickness)
        return image
    def put_index(self, image, stridx, ratio = 1):
        FONT, FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 1, 1
        (label_width, label_height), baseline = cv2.getTextSize(str(stridx), FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.putText(image, str(stridx),(int(self.center[0]*ratio - label_width / 2),
                     int(self.center[1]*ratio + label_height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    def fill(self, image, ratio):
        start_x, start_y = int((self.center[0] - self.width / 2)*ratio), int((self.center[1] - self.height / 2)*ratio)
        sor = np.ones((int(self.height*ratio), int(self.width*ratio), 3), np.uint8) * 255  # square of regard
        image[start_y:start_y + sor.shape[0], start_x:start_x + sor.shape[1]] = sor
        return image

class board:
    def __init__(self):
        self.num = 6 # 6x6 grid
        self.width = 1020  # mm
        self.height = 1380  # mm
        self.grid_cell = []
        self.cam_loc = (self.width/2, 550)#254) # location of camera on board
        for idx in range(0, self.num**2):
            cell_idx = cell()
            cell_idx.center = (cell_idx.width * (idx % self.num) + cell_idx.width / 2,
                           cell_idx.height * int(idx / self.num) + cell_idx.height / 2)
            self.grid_cell.append(cell_idx)
        # initiate to display virtual board on screen with ratio
        display = Gdk.Display.get_default()
        screen = display.get_default_screen()
        default_screen = screen.get_default()
        screen_height = default_screen.get_height()
        self.create_visual(screen_height)

    def create_visual(self, monitor_height, ratio = 0.6): # ratio on monitor height
        h_pixels = int(ratio * monitor_height)
        w_pixels = int(ratio * monitor_height * (self.width / self.height))  # 102 / 138
        background = np.zeros((h_pixels, w_pixels, 3))  # 3channel
        ratio = (h_pixels / self.height)
        for idx in range(0, len(self.grid_cell)):
            self.grid_cell[idx].draw(background, ratio) # 1 mm = h/1380
            self.grid_cell[idx].put_index(background, idx+1, ratio)
        self.background = background
        self.ratio = ratio

    def visual_ror(self, idx):
        image = self.background.copy()
        cam_x = self.cam_loc[0]-self.grid_cell[idx].center[0]
        cam_y = self.grid_cell[idx].center[1]-self.cam_loc[1]
        return self.grid_cell[idx].fill(image, self.ratio), (cam_x, cam_y)

    def visual_ror_xy(self, x_hat, y_hat): # region of regard
        cell_x = self.cam_loc[0] - x_hat
        cell_y = self.cam_loc[1] + y_hat
        w, h = self.grid_cell[0].width, self.grid_cell[0].height
        cell_fill = int(cell_x/w) + (int(cell_y/h))*6

        if cell_fill < 0 or cell_fill >= self.num**2:
            img = self.background.copy()
            text = 'Out of View'
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 1
            (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.putText(img, text,(int((img.shape[1] - label_width) / 2), int((img.shape[0] - label_height) / 2)),
                        font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            return img, None
        return self.visual_ror(cell_fill)

    def visual_por_xy(self, x_hat, y_hat): # point of regard
        img, _ = self.visual_ror_xy(x_hat, y_hat)
        cell_x = int((self.cam_loc[0] - x_hat)*self.ratio)
        cell_y = int((self.cam_loc[1] + y_hat)*self.ratio)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '.', (cell_x, cell_y), font, 3, (58, 245, 28), 10, cv2.LINE_AA)

        return img

    def visual_por_xy_finetune(self, g_t, g_h):
        x_t, y_t = g_t
        x_hat, y_hat = g_h
        img, _ = self.visual_ror_xy(x_hat, y_hat)
        cell_x = int((self.cam_loc[0] - x_t)*self.ratio)
        cell_y = int((self.cam_loc[1] + y_t)*self.ratio)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '.', (cell_x, cell_y), font, 3, (255, 255, 255), 10, cv2.LINE_AA) # White for ground truth
        cell_x = int((self.cam_loc[0] - x_hat) * self.ratio)
        cell_y = int((self.cam_loc[1] + y_hat) * self.ratio)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '.', (cell_x, cell_y), font, 3, (58, 245, 28), 10, cv2.LINE_AA)

        return img

    def visual_gt(self, img, g_tx, g_ty): # visual ground truth
        cell_x = int((self.cam_loc[0] - g_tx)*self.ratio)
        cell_y = int((self.cam_loc[1] + g_ty)*self.ratio)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '.', (cell_x, cell_y), font, 3, (0, 0, 28), 10, cv2.LINE_AA)

        return img


def grab_img(cap):
    global THREAD_RUNNING
    global frames
    while THREAD_RUNNING:
        frame = cap.wait_for_frames(timeout_ms=10000) #_, frame = cap.read()
        frame = frame.get_color_frame()
        frame = np.asanyarray(frame.get_data())
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frames.append(frame)

def collect_data(cap):
    global THREAD_RUNNING
    global frames

    virtualboard = board()

    calib_data = {'frames': [], 'g_t': []}
    i = 0
    while i < virtualboard.num**2:
        # Start the sub-thread, which is responsible for grabbing images
        frames = []
        THREAD_RUNNING = True
        th = threading.Thread(target=grab_img, args=(cap,))
        th.start()
        img, g_t = virtualboard.visual_ror(i)
        cv2.imshow('image', img)
        key_press = cv2.waitKey(0)
        if key_press == 32: # space
            print(i, g_t)
            THREAD_RUNNING = False
            th.join()
            calib_data['frames'].append(frames)
            calib_data['g_t'].append(g_t)
            i += 1
        elif key_press & 0xFF == ord('q'):
            THREAD_RUNNING = False
            cv2.destroyAllWindows()
            break
        else:
            THREAD_RUNNING = False
            th.join()

    return calib_data

def save_dataset(data):
    name = input('Enter your name: ')
    now = datetime.datetime.now()
    subject = name + '_' + now.strftime("%Y-%m-%d %H.%M.%S")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('%s_calib.avi' % subject, fourcc, 30.0, (1280, 720))
    target = []
    for idx in range(0, len(data['g_t'])):
        frames = data['frames'][idx]
        g_t = data['g_t'][idx]
        for i in range(len(frames) - 10, len(frames)):
            frame = frames[i]
            target.append(g_t)
            out.write(frame)

    out.release()
    fout = open('%s_calib_target.pkl' % subject, 'wb')
    pickle.dump(target, fout)
    fout.close()
    print('saved')

if __name__ == "__main__":
    import pyrealsense2 as rs
    from subprocess import call
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(config)
    cam_idx = 3
    # adjust these for your camera to get the best accuracy
    call('v4l2-ctl -d /dev/video%d -c brightness=100' % cam_idx, shell=True)
    call('v4l2-ctl -d /dev/video%d -c contrast=50' % cam_idx, shell=True)
    call('v4l2-ctl -d /dev/video%d -c sharpness=100' % cam_idx, shell=True)
    
    data = collect_data(pipe)
    if len(data['g_t']) == 36:
        save_dataset(data)
    
    pipe.stop()
