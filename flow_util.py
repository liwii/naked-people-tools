import numpy as np
import cv2
import skvideo.io

def get_flow(filepath: str):
    # Get the optical flow using OpenCV
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

    # flow.shape (2, frame_count, height, width)
    # flow[0] is dx, flow[1] is dy
    # Use TV-L1 algorithm instead, since it is used in mlb-youtube dataset
    # https://www.ipol.im/pub/art/2013/26/article.pdf

    cap = cv2.VideoCapture(filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame1 = cap.read()
    prvf = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow_video = []
    for i in range(length - 1):
        success, frame2 = cap.read()
        nextf = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if not success:
            break
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        flow_frame = optical_flow.calc(prvf, nextf, None)
        # flow_frame = cv2.calcOpticalFlowFarneback(prvf,nextf, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_video.append(flow_frame)
        prvf = nextf
    cap.release()
    flow_video = np.array(flow_video)
    return flow_video.transpose(3, 0, 1, 2)

def save_flow(flow, output_file):
    _, frame_count, height, width = flow.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    writer = skvideo.io.FFmpegWriter(output_file,
                                outputdict={"-vcodec":"libx264", "-pix_fmt": "yuv420p"})
    for i in range(frame_count):
        flow_frame = flow[:, i, :, :]
        mag, ang = cv2.cartToPolar(flow_frame[0], flow_frame[1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        writer.writeFrame(rgb)
    writer.close()