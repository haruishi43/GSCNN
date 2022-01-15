#!/usr/bin/env python3

import argparse

from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='path to config `.py` file'
    )
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='argument in dict',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    print(">>> Config:")
    print(cfg.pretty_text)


if __name__ == "__main__":
    main()
