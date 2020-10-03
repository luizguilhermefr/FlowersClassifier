import argparse


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path of the input image", type=str)
    parser.add_argument("model", help="Path of the model to use", type=str)
    parser.add_argument(
        "--top_k",
        help="Number of results to return (most likely K results)",
        default=5,
        type=int,
    )

    return parser
