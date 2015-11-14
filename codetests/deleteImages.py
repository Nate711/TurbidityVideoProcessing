import os, glob
files = (os.listdir(os.getcwd()))
print files


for name in files:
	if(int(name[17:-4])%100 != 0):
		os.remove(name)

#for filename in gl-ob.glob("version*"):
#    os.remove(filename) 