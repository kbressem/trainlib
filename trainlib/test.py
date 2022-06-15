import os
import shutil
dir_name = os.path.dirname(os.path.abspath(__file__))
shutil.copytree(dir_name, "/home/bressekk/inpaint/runs")