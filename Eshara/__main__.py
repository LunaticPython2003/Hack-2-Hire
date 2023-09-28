import os
import sys
os.chdir('Eshara')
# os.system("pip install -r requirements.txt")
if sys.platform=="darwin":
    os.system('python3 main.py')
else:
    os.system('python main.py')