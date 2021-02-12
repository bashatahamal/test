import subprocess
import time


# def inner():
proc = subprocess.Popen(
    # call something with a lot of output so we can see it
    ["python", "-u", "count_timer.py"],
    stdout=subprocess.PIPE,
    # universal_newlines=True
)

for line in iter(proc.stdout.readline, b''):
    # print(line.decode("utf-8"))
    print(line.decode("utf-8").rstrip() + '<br/>\n')
# for line in proc.stdout:
#     print(line)

#     for line in iter(proc.stdout.readline, ''):
#         print(line)
#         # Don't need this just shows the text streaming
#         time.sleep(1)
#         yield line.rstrip() + '<br/>\n'

# inner()

# def execute(cmd):
#     popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
#     for stdout_line in iter(popen.stdout.readline, b''):
#         yield stdout_line
#     popen.stdout.close()
#     return_code = popen.wait()
#     if return_code:
#         raise subprocess.CalledProcessError(return_code, cmd)

# for path in execute(["python", "-u", "count_timer.py"]):
#     print(path, end="")
#     break
# execute(["python", "-u", "count_timer.py"])
# import sys
# def execute(cmd):
#     # popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
#     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     while True:
#         nextline = process.stdout.readline()
#         if nextline == '' and process.poll() is not None:
#             break
#         sys.stdout.write(nextline)
#         sys.stdout.flush()
    
#     output = process.communicate()[0]
#     exitCode = process.returncode

#     if (exitCode == 0):
#         return output
#     else:
#         raise ProcessException(command, exitCode, output)

# execute(["python count_timer.py"])
   