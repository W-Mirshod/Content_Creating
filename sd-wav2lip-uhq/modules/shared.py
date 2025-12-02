"""
Minimal stub implementation of modules.shared for standalone sd-wav2lip-uhq usage.
This replaces the Stable Diffusion WebUI shared module.
"""
from types import SimpleNamespace


class State:
    """Stub state object for tracking processing state"""
    def __init__(self):
        self.interrupted = False
    
    def interrupt(self):
        """Set interrupted flag"""
        self.interrupted = True
    
    def begin(self):
        """Reset interrupted flag"""
        self.interrupted = False


class CmdOpts:
    """Stub command line options object"""
    def __init__(self):
        self.disable_safe_unpickle = False


class Opts:
    """Stub options object for face restoration settings"""
    def __init__(self):
        self.code_former_weight = 0.5
        self.face_restoration_model = "CodeFormer"


# Global instances
state = State()
cmd_opts = CmdOpts()
opts = Opts()

