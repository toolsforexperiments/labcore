import os
import argparse

def run_app():
    # Called by the 'Rocky' entry point
    print("RUNNING APP IN RUN_APP!")
    parser = argparse.ArgumentParser(description="A data monitoring website created for the Pfaff Quantum Circuit Lab by Rocky Daehler,\n based on the Plottr tool by wolfgang Pfaff.")
    parser.add_argument('-f', '--file', type=str, help='An input string')

    args = parser.parse_args()
    print(args)
    
    os.system('panel serve scripts\monitr_server.py')

if __name__ == "__main__":
    print("Hello")
    run_app()