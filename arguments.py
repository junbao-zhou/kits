import argparse
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=config.LOG_DIR,
        required=False,
        help='Directory to put the log data. Default: ~/logs/date+time'
    )
    args, unparsed = parser.parse_known_args()
    return args, unparsed