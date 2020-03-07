"""Some simple regularization functions."""

class l2:
    def __init__(self, alpha=1e-4) -> None:
        self.alpha = alpha