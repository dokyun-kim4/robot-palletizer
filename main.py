from backend.kinova import Kinova
import sys, time
import numpy as np
import queue, threading


class Main:

    def __init__(self, loop_rate=20) -> None:
        self.kinova_robot = Kinova()
        self.LOOP_RATE = 1 / float(loop_rate)

        self.action_queue = queue.Queue()

        self.is_running = True

        self.start()

        self.background_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.background_thread.start()
        print("Loop Started")

    # DO NOT TOUCH
    def _start_loop(self):
        try:
            while self.is_running:
                if not self.action_queue.empty():
                    func, args = self.action_queue.get()
                    print(f"Executing: {func.__name__}")
                    func(*args)
                self.loop()
                time.sleep(self.LOOP_RATE)
        except Exception as e:
            print(f"ERROR Background loop crashed: {e}")

    # DO NOT TOUCH
    def shutdown(self):
        print("Shutting down gracefully")
        self.is_running = False
        self.kinova_robot.set_torque(True)
        self.kinova_robot.stop()
        sys.exit(0)

    def start(self):
        # PUT START CODE IN HERE
        pass

    def loop(self):
        # PUT PERIODIC CODE IN HERE
        pass


if __name__ == "__main__":
    final_project = Main()
    # KEEP THIS LOOP
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()
