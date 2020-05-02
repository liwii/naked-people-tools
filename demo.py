from naked_people_generator import NakedPeopleGenarator
from flow_util import get_flow, save_flow
from math import pi
bmlmovi_path = '/Users/ryuukouki/Princeton/ACV/BMLmovi/'
smplh_path = '/Users/ryuukouki/Princeton/ACV/smplh'
dmpl_path = '/Users/ryuukouki/Princeton/ACV/dmpls'

generator = NakedPeopleGenarator(bmlmovi_path, smplh_path, dmpl_path)

generator.nakedgen('naked.mp4', 56, 1, rotation=pi/6, bg_image='room.jpeg', color='grey', bg_color='white', imw=400, imh=400, frame_skip=4)

flow_np = get_flow('naked.mp4')

save_flow(flow_np, 'naked_flow.mp4')
