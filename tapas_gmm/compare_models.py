from argparse import ArgumentParser

import torch

from tapas_gmm.utils.torch import compare_state_dicts


def main(path_a, path_b):
    state_a = torch.load(path_a, map_location=torch.device("cpu"))
    state_b = torch.load(path_b, map_location=torch.device("cpu"))

    return compare_state_dicts(state_a, state_b)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", dest="a", required=True, help="Path of the first model.")
    parser.add_argument("-b", dest="b", required=True, help="Path of the second model.")
    args = parser.parse_args()

    main(args.a, args.b)
