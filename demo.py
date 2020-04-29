from naked_people_generator import NakedPeopleGenarator
from math import pi
bmlmovi_path = '/Users/ryuukouki/Princeton/ACV/BMLmovi/'
smplh_path = '/Users/ryuukouki/Princeton/ACV/smplh'
dmpl_path = '/Users/ryuukouki/Princeton/ACV/dmpls'

generator = NakedPeopleGenarator(bmlmovi_path, smplh_path, dmpl_path)

generator.nakedgen('naked.mp4', 1, 1, rotation=pi/3, color='green', bgcolor='yellow', imw=400, imh=400, frame_skip=4)
