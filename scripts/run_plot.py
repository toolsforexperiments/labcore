import os

def run_app():
    # Called by the 'Rocky' entry point
    os.system('panel serve scripts\monitr_server.py')

if __name__ == "__main__":
    print("Hello")
    run_app()