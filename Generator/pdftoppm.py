import os
import subprocess
import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--pdf", required=True,
	            help="PDF File Location")
# ap.add_argument("-c", "--count", help="Number of character in PDF")
args = vars(ap.parse_args())
pdf_location = args["pdf"]
# Get font type
slash = pdf_location.split('/')
for k in slash:
    file_name = k.split('^')
    if len(file_name) > 1:
        for name in file_name:
            font_name = name.split('.')
            if len(font_name) > 1:
                font_type = font_name[0]
def runCommand(cmd, timeout=None, window=None):
	p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	output = ''
	for line in p.stdout:
		line = line.decode(errors='replace' if (sys.version_info) < (3, 5) else 'backslashreplace').rstrip()
		output += line
		# print(line)
	retval = p.wait(timeout)
	return (retval, output)


store_folder = '/home/mhbrt/Desktop/Wind/Multiscale/Generator/Form A/' + font_type
if not os.path.exists(store_folder):
    os.makedirs(store_folder)
pdftoppm = 'pdftoppm "' +pdf_location+ '" "' + store_folder +'/'+font_type+ '" -png -rx 300 -ry 300'

print('converting...')
_,ret = runCommand(pdftoppm)
# print(ret)
print('Done')