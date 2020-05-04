import json
import random
import numpy as np
from math import pi
from flow_util import get_flow, save_flow

from naked_people_generator import NakedPeopleGenarator
images = ["room.jpg", "room2.jpg", "room3.jpg", "field.jpg", "field2.jpg", "field3.jpg"]
actions = ["walking", "hand_waving", "sitting_down"]

with open("amass_metadata.json") as f:
    subjects = json.load(f)["subjects"]

bmlmovi_path = '/Users/ryuukouki/Princeton/ACV/BMLmovi/'
smplh_path = '/Users/ryuukouki/Princeton/ACV/smplh'
dmpl_path = '/Users/ryuukouki/Princeton/ACV/dmpls'

generator = NakedPeopleGenarator(bmlmovi_path, smplh_path, dmpl_path)

imw, imh = 256, 256

for i, s in enumerate(subjects):
    idx = i + 1
    if idx > 20:
        break
    if "moves" not in s:
        continue
    moves = s["moves"]
    for j, m in enumerate(moves):
        if m not in actions:
            continue
        jdx = j + 1
        for n in range(0, 10):
            ndx = n + 1
            video_name = "subject_{}_move_{}_{}".format(idx, jdx, ndx)
            videofile = "videos/{}/{}.mp4".format(m, video_name)
            flowfile = "flows/{}/{}_flow".format(m, video_name)
            image = random.choice(images)
            rotation = random.uniform(-pi/2, pi/2)
            generator.nakedgen(videofile, idx, jdx, imw=imw, imh=imh, bg_image=image, rotation=rotation, frame_skip=5)

            # If you want to generate flows at the same time and save them as npz, please uncomment the following lines.
            #flow_np = get_flow(videofile)
            #np.save(flowfile, flow_np)
            print(video_name)