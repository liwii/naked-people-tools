import json
import random
import numpy as np
from math import pi
from flow_util import get_flow, save_flow

from naked_people_generator import NakedPeopleGenarator
images = ["room.jpg", "room2.jpg", "room3.jpg", "field.jpg", "field2.jpg", "field3.jpg"]
actions = ["walking", "hand_waving", "sitting_down"]
colors = ['pink', 'purple', 'cyan', 'red', 'green', 'yellow', 'brown', 'blue', 'offwhite', 'white', 'orange', 'grey', 'yellowg']

with open("amass_metadata.json") as f:
    subjects = json.load(f)["subjects"]

bmlmovi_path = '/Users/ryuukouki/Princeton/ACV/BMLmovi/'
smplh_path = '/Users/ryuukouki/Princeton/ACV/smplh'
dmpl_path = '/Users/ryuukouki/Princeton/ACV/dmpls'

generator = NakedPeopleGenarator(bmlmovi_path, smplh_path, dmpl_path)

imw, imh = 224, 224

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
            bg_color = random.choice(colors)
            rotation = random.uniform(-pi/2, pi/2)
            rotation_to = random.uniform(-pi/2, pi/2)
            #scale = random.uniform(0.7, 1.3)
            transx, transy = random.uniform(-0.5, 0.5), random.uniform(-0.1, 0.1)
            transx_to, transy_to = random.uniform(-0.5, 0.5), random.uniform(-0.1, 0.1)
            generator.nakedgen(videofile, idx, jdx, imw=imw, imh=imh, bg_image=image, rotation=rotation, rotate_to=rotation_to, frame_skip=5, bg_color=bg_color, translation=[transx, transy], translation_to=[transx_to, transy_to])

            # If you want to generate flows at the same time and save them as npz, please uncomment the following lines.
            #flow_np = get_flow(videofile)
            #np.save(flowfile, flow_np)
            print(video_name)