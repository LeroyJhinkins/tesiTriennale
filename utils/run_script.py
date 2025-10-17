import libs.SignalHandler
import os
import sys
import subprocess

temp_file_path = "temp/run_memory.txt"

def main():
    if len(sys.argv) == 1: # only run was used
        # first check if the temp file exists AND the saved script name still exists
        if not os.path.exists(temp_file_path):
            print("[Error] failed to get the previous command. Usage: run <script> [args...]")
            sys.exit(1)
        # second, read the file
        with open(temp_file_path, "r") as file:
            script_name = file.read()
    if len(sys.argv) >= 2: # run <script> was passed
        script_name = sys.argv[1]

    script_path = f"scripts/{script_name}.py"
    # verify script existance
    if not os.path.exists(script_path):
        print(f"[Error] the script \"{script_path}\" doesn't exist or was deleted")
        sys.exit(1)

    # save verified script (overwrite)
    with open(temp_file_path, "w") as file:
        script_name = file.write(script_name)
    
    # launch and pass remaining arguments to the script
    subprocess.run([sys.executable, script_path] + sys.argv[2:])

