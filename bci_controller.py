import time


class BCIController:
    """
    Placeholder controller for Flappy Bird.

    Right now this class does not connect to OpenBCI or run a classifier.
    It simply provides the interface your game will use later.

    When your classifier is ready, you can replace the inside of should_jump()
    with live EEG reading + preprocessing + model prediction.
    """

    def __init__(self):
        self.enabled = False
        self.last_jump_time = 0.0
        self.cooldown_seconds = 0.35

    def start(self):
        """Set up any live resources here later."""
        self.enabled = True

    def stop(self):
        """Clean up any live resources here later."""
        self.enabled = False

    def should_jump(self):
        """
        Return True when the bird should flap.

        Placeholder behavior:
        always returns False.

        Later this method can:
        1. Read the newest EEG samples from the Cyton
        2. Preprocess a short rolling window
        3. Run the classifier
        4. Apply a cooldown so one blink does not trigger many jumps
        5. Return True only when a jump should happen
        """
        if not self.enabled:
            return False

        detected = False

        if detected and self._cooldown_ready():
            self.last_jump_time = time.time()
            return True

        return False

    def _cooldown_ready(self):
        return (time.time() - self.last_jump_time) >= self.cooldown_seconds
