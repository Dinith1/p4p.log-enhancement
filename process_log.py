import argparse
import glove.glove as glove

if __name__ == '__main__':
    # Set up CLI arguments
    parser = argparse.ArgumentParser(description='Process a log file to make it simpler.')

    parser.add_argument('--log', '-l',
                        action='store',
                        required=True,
                        help='The filename of the log to be processed'
                        )

    parser.add_argument('--model', '-m',
                        action='store',
                        required=True,
                        help='The filename of the pre-trained GloVe model to use'
                        )

    args = parser.parse_args()

    log = args.log.replace("\\", "/")
    model = args.model.replace("\\", "/")

    words = glove.process_log(log, model)

    print(words)
