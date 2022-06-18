import glob, os
from pathlib import Path
for file in glob.glob("**/IM*", recursive = True):
    os.rename(file,file+".dcm")

