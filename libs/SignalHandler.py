import signal
import sys

class SignalHandler:
    def __init__(self, cleanup_func=None):
        """
        Initialize signal handling.
        :param cleanup_func: Function to call on signal (e.g., cleanup before exit)
        """
        self.cleanup_func = cleanup_func
        signal.signal(signal.SIGINT, self.handle_exit)   # Ctrl+C
        signal.signal(signal.SIGTERM, self.handle_exit)  # Kill command

    def handle_exit(self, signum, frame):
        """ Handle termination signals. """
        print(f"\n[SignalHandler] Caught signal {signum}, exiting...")
        if self.cleanup_func:
            self.cleanup_func()
        sys.exit(0)

# Create a global instance to activate it
default_handler = SignalHandler()
