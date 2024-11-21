from argparse import ArgumentParser
from pprint import pformat
from textwrap import indent

import numpy as np
import torch
import viz.action_distribution as action_distribution

from tapas_gmm.utils.debug import summarize_list_of_tensors, summarize_tensor
from tapas_gmm.utils.misc import load_replay_memory

PRINTABLE = "printable"
SUMMARIZABLE = "summarizable"
OTHER = "other"


def tensor_printable(tensor, max_elements=25):
    if tensor.nelement() > max_elements:
        return False
    else:
        return True


def list_printable(obj, max_elements=25):
    if len(obj) > max_elements:
        return False
    else:
        return True


def summarize(obj):
    if type(obj) == list:
        # HACK TODO: make generic
        is_image_like = obj[0].shape[-1] in (128, 256)
        return summarize_list_of_tensors(obj, keep_last_dim=not is_image_like)
    elif type(obj) == torch.Tensor:
        return summarize_tensor(obj, "...")
    else:
        raise ValueError("Unexpected type to summarize {}".format(type(obj)))


def desc_type(obj):
    if type(obj) == list:
        return "[" + ", ".join([str(type(i)) for i in obj]) + "]"
    else:
        return str(type(obj))


def get_print_type(obj):
    t = type(obj)
    if t == list:
        if not obj:  # empty list
            return PRINTABLE
        if not all(isinstance(i, type(obj[0])) for i in obj):
            # list elements are not of same type
            return OTHER
        else:  # recurse on objects
            return get_print_type(obj[0]) if list_printable(obj) else SUMMARIZABLE
    elif t == torch.Tensor:
        if tensor_printable(obj):  # summarize if large, else print
            return PRINTABLE
        else:
            return SUMMARIZABLE
    else:
        return PRINTABLE


def show_action_distribution(replay_memory):
    action_store = np.concatenate([t.numpy() for t in replay_memory.action])
    action_store = np.swapaxes(action_store, 0, 1)
    print(action_store.shape)  # should be (7, n_traj*len_traj)
    action_distribution.make_all(action_store)


def main(replay_memory):
    attr_dict = vars(replay_memory)

    for k, v in attr_dict.items():
        if (ptype := get_print_type(v)) == PRINTABLE:
            print(k)
            print(indent(pformat(v), "    "))
        elif ptype == SUMMARIZABLE:
            print(k)
            print(indent(pformat(summarize(v)), "   "))
        elif ptype == OTHER:
            print(k)
            print(indent(pformat(desc_type(v)), "    "))

    # print(sorted(list(set().union(*[t.int().unique().tolist()
    #                                 for t in replay_memory.masks_o]))))

    show_action_distribution(replay_memory)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning_10",
        help="options: cloning_10, cloning_200",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        # help="options: {}, 'Mixed'".format(str(tasks)[1:-1]),
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        action="store_true",
        default=False,
        help="Use data with ground truth object masks.",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        default=None,
        help="Path to a dataset. May be provided instead of f-t-m.",
    )
    args = parser.parse_args()
    config = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "ground_truth_mask": args.mask,
    }
    if args.path:
        replay_memory = torch.load(args.path)
    else:
        replay_memory = load_replay_memory(config)
    main(replay_memory)
