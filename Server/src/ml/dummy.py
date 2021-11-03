import time
import sys

def main():
    sec = 0
    print("start main")
    while sec < 500:
        time.sleep(1)
        sys.stdout.write("sec: {sec}")
        sys.stdout.flush()
        sec += 1

if __name__ == "__main__":
    main()