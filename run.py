import argparse
from lib.cli.sessions import Static_Session

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='choose between optimization, validation, or train.')
  parser.add_argument('-n', '--name', type=str, required=True,
                      help='Name for the session"')

  parser.add_argument('-e', '--episodes', type=int, default=1000,
                      help='number of episodes to run')
  parser.add_argument('-i', '--initial_invest', type=int, default=1000,
                      help='initial investment amount')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')

  # Parses arguments
  args = parser.parse_args()
  
  if args.mode == "train" or args.mode == "validation" or args.mode == "optimization":
    s = Static_Session(args.mode, args.weights, args.episodes, args.initial_invest, args.name)
    
  if args.mode == "live":
    raise NotImplementedError
