from typing import List

import numpy as np
import pickle
import os


class Selector:
    def __init__(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.mask_code = pickle.load(
            open(os.path.join(root_path, "mask_code.pkl"), "rb")
        )

    def get_mask_code(self) -> List[int]:
        """
        Returns: The indices of the 30 features.
        """
        if self.mask_code is None:
            raise ValueError(
                "Mask code is not loaded or set. Please load the mask code."
            )
        return self.mask_code.tolist()


if __name__ == "__main__":
    selector = Selector()
    mask_code = selector.get_mask_code()
    print("Mask code (selected features):", mask_code)
